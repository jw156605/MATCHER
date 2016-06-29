import numpy as np
import matplotlib.pyplot as plt
import GPy as gpy
from scipy.stats import chi2
from scipy.stats import spearmanr

import sys

class MATCHER:
    """ Manifold Alignment to Characterize Experimental Relationships (MATCHER)
        
        :param X: One or more single cell datasets. To learn a joint trajectory from two datasets with known correspondences, add a second level list.
        :type X: list, possibly nested
        :param missing_data: Model missing data?
        :type missing_data: bool
        
    """
    def __init__(self,X,missing_data=True):
        """ Constructor for MATCHER class """
        self.X = X
        self.model = []
        
        for dataset in self.X:
            num_inducing = min(100,dataset.shape[1])
            if isinstance(dataset,list):
                self.model.append(gpy.models.mrd.MRD(dataset, 1, num_inducing=num_inducing, kernel=gpy.kern.RBF(1)))
            else:
                self.model.append(gpy.models.bayesian_gplvm.BayesianGPLVM(dataset, 1, num_inducing=num_inducing, missing_data=missing_data, kernel=gpy.kern.RBF(1)))
        self.warp = []
        self.pseudotime = []
        self.quantiles = -1

    def infer(self,quantiles=10):
        """ Infer pseudotime values using variational Bayes.
            
            :param quantiles: How many quantiles to use when computing warp functions
            :type quantiles: int
            
        """
        self.warp = []
        self.pseudotime = []
        self.quantiles = quantiles
        for i in range(len(self.X)):
            self.model[i].optimize(messages=True)
            t = self.model[i].latent_space.mean
            x = np.percentile(t,range(0,100+100/quantiles,quantiles)).reshape(quantiles+1,1)
            y = np.arange(0,1.+1./quantiles,1./quantiles).reshape(quantiles+1,1)
            self.warp.append(gpy.models.GPRegression(x,y,gpy.kern.RBF(1)))
            self.warp[i].optimize(messages=True)
            self.pseudotime.append(self.warp[i].predict(t)[0])

    def reverse_pseudotime(self,inds):
        """ Reverse the direction of pseudotime for the specified model(s).
            This handles the case when the trajectories from different data types move in opposite directions.
        
            :param inds: Indices of the model(s) to reverse
            :type inds: list
            
        """
        for i in inds:
            t = self.model[i].latent_space.mean
            x = np.percentile(t,range(0,100+100/self.quantiles,self.quantiles)).reshape(self.quantiles+1,1)
            y = np.arange(1.,-1./self.quantiles,-1./self.quantiles).reshape(self.quantiles+1,1)
            self.warp[i] = gpy.models.GPRegression(x,y,gpy.kern.RBF(1))
            self.warp[i].optimize(messages=True)
            self.pseudotime[i] = self.warp[i].predict(t)[0]
            
    def sample_pseudotime(self, ind, samples=10):
        """ Sample from the posterior for the inferred pseudotime values for the specified model.
        
            :param ind: Index of model to sample
            :type ind: int
            :param samples: Number of samples
            :type samples: int
            :returns: Posterior samples from inferred pseudotime values for each cell
            :rtype: SxN array, where S is the number of samples and N is the number of cells
        
        """
        means = self.model[ind].latent_space.mean.flatten()
        variances = self.model[ind].latent_space.variance.flatten()
        sampled = np.random.multivariate_normal(means,np.diag(variances),samples)
        mapped = self.warp[ind].predict(sampled.reshape(-1,1))[0].reshape(samples,-1)
        return mapped
            
    # def find_markers(self,inds):
    #     p_vals = []
    #     for ind in inds:
    #         num_cells, num_features = self.X[ind].shape
    #         raw_p_vals = []
    #         for j in range(num_features):
    #             t = self.model[ind].latent_space.mean
    #             feature = self.X[ind][range(num_cells),j].reshape(-1,1)
    #             # Fit GP model with constant (bias) kernel
    #             null_model = gpy.models.GPRegression(t,feature,gpy.kern.Bias(1))
    #             null_model.optimize()        
    #             # Fit GP model with RBF kernel
    #             alt_model = gpy.models.GPRegression(t,feature,gpy.kern.RBF(1))
    #             alt_model.optimize()
    #             df = len(alt_model.parameter_names_flat()) - len(null_model.parameter_names_flat())
    #             # Compute likelihood ratio statistic using chisquare with one df
    #             D = 2*(alt_model.log_likelihood() - null_model.log_likelihood())
    #             p = chi2.sf(D,df)
    #             raw_p_vals.append(p)
    #             if (j+1 % 100) == 0:
    #                 print str(j+1) + "/" + str(num_features) + " features tested"
    #                 sys.stdout.flush()
    #         p_vals.append([x * num_features for x in raw_p_vals])
    #     return p_vals
    
    def _p_adjust(self,p):
        """ Helper function to perform Benjamini-Hochberg FDR control.
        
            :param p: p-values to adjust
            :type p: array
            :returns: adjusted p-values
            :rtype: list of floats
        
        """
        p = np.asfarray(p)
        by_descend = p.argsort()[::-1]
        by_orig = by_descend.argsort()
        steps = float(len(p)) / np.arange(len(p), 0, -1)
        q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
        return q[by_orig]
    
    def find_markers(self,inds,fdr_correction=True):
        """ Find features that are significantly related to pseudotime from each specified data type.
            
            :param inds: Indices of data types to use
            :type inds: list
            :param fdr_correction: Perform multiple hypothesis test correction to control FDR?
            :type fdr_correction: bool
            :returns: list of indices for each data type
            :rtype: list of lists
            
        """
        p_vals = []
        for ind in inds:
            num_cells, num_features = self.X[ind].shape
            raw_p_vals = []
            t = self.model[ind].latent_space.mean
            F = self.model[ind].predict(t)[0]
            # Fit GP model with constant (bias) kernel
            null_model = gpy.models.GPRegression(t,self.X[ind],gpy.kern.Bias(1))
            null_model.optimize()
            F_null = null_model.predict(t)[0]
            for j in range(num_features):
                f = F[:,j]
                f_null = F_null[:,j]
                y = self.X[ind][:,j]
                null_loglik = sum(null_model.likelihood.logpdf_link(f_null,y))        
                alt_loglik = sum(self.model[ind].likelihood.logpdf_link(f,y))
                df = 1
                # Compute likelihood ratio statistic using chisquare with one df
                D = 2*(alt_loglik - null_loglik)
                p = chi2.sf(D,df)
                raw_p_vals.append(p)
            if fdr_correction:
                p_vals.append(self._p_adjust(raw_p_vals))
            else:
                p_vals.append(raw_p_vals)
        return p_vals
    
    def correlation(self,model1,model2,inds1,inds2):
        """ Approximate correlation between the specified features of two different data types.
        
            :param model1: Index of first data type
            :type model1: int
            :param model2: Index of second data type
            :type model2: int
            :param inds1: Indices of features
            :type inds1: list
            :param inds2: Indices of features
            :type inds2: list
            :returns: Spearman correlation matrix and p-values
            :rytpe: list of arrays
            
        """
        t = self.pseudotime[model1]
        model2_internal = self.warp[model2].infer_newX(t.reshape(-1,1))[0]
        vals1 = self.X[model1][:,inds1]
        vals2 = self.model[model2].predict(model2_internal)[0][:,inds2]
        corr_res = spearmanr(vals1,vals2)
        n1 = len(inds1)
        n2 = len(inds2)
        n = n1 + n2
        corr_mat = corr_res[0][range(n1),n2:n]
        p_vals = corr_res[1][range(n1),n2:n]
        return [corr_mat,p_vals]
        
    def plot_warping_functions(self,inds):
        """ Plot the functions for each data type that map from "domain-specific pseudotime" to "master pseudotime".
        
            :param inds: Indices of data types
            :type inds: list
        
        """
        fig, axes = plt.subplots(1,len(inds))
        for i in range(len(inds)):
            self.warp[inds[i]].plot(ax=axes[i])
            
    def plot_feature(self,model_ind,feature_ind):
        """ Plot the specified feature and its model fit
        
            :param model_ind: Index for data type
            :type model_ind: int
            :param feature_ind: Index of feature
            :type feature_ind: int
        
        """
        fig = self.model[model_ind].plot_f(which_data_ycols=[feature_ind],plot_inducing=False)
        fig.scatter(self.model[model_ind].latent_space.mean,self.X[model_ind][:,feature_ind],marker='x',c="black")
        
    def plot_pseudotime(self,inds):
        """ Plot inferred pseudotime values and posterior uncertainty for models specified by inds.
        
            :param inds: Indices of data types
            :type inds: list
        
        """
        fig, axes = plt.subplots(1,len(inds))
        lines = []
        fills = []
        bg_lines = []
        for i in range(len(self.model)):
            means = self.model[i].latent_space.mean.flatten()
            x = range(len(means))
            vars = self.model[i].latent_space.variance.flatten()
            axis_order = np.argsort(means)
            ci_up = means[axis_order] + 2 * np.sqrt(vars[axis_order])
            ci_up = self.warp[i].predict(ci_up.reshape(-1,1))[0].flatten()
            ci_down = means[axis_order] - 2 * np.sqrt(vars[axis_order])
            ci_down = self.warp[i].predict(ci_down.reshape(-1,1))[0].flatten()
            means = self.warp[i].predict(means.reshape(-1,1))[0].flatten()
            lines.extend(axes[i].plot(x, means[axis_order], c='b', label=r"$\mathbf{{T_{{{}}}}}$".format(i+1)))
            fills.append(axes[i].fill_between(x, ci_down, ci_up, facecolor=lines[i].get_color(), alpha=.3))
            axes[i].legend(borderaxespad=0.,loc=0)
            axes[i].set_xlim(min(x), max(x))
            axes[i].set_ylim(min(means), max(means))
        plt.draw()
        axes[len(self.model)-1].figure.tight_layout(h_pad=.01)
        return dict(lines=lines, fills=fills, bg_lines=bg_lines)
