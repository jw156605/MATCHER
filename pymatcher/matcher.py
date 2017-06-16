import numpy as np
import matplotlib.pyplot as plt
import GPy as gpy
from scipy.stats import chi2
from scipy.stats import spearmanr
from scipy.interpolate import interp1d
import sys

class WarpFunction:
    """ Class to learn a (possibly nonlinear) function warping from
        observed distribution to uniform (0,1) distribution and back.
        
        :param t: pseudotime values
        :type t: Nx1 array of float
        :param quantiles: Number of quantiles to use in learning warp
        :type quantiles: integer
        :param method: Gaussian process regression or linear interpolation?
        :type method: string
        :param reverse: Reverse uniform quantiles before warping?
        :type reverse: boolean
        
    """
    def __init__(self,t,quantiles=50,method='gp',reverse=False):
        """ Constructor for MATCHER class """
        self.method = method
        self.quantiles = quantiles
        self.reverse = reverse
        N = t.shape[0]
        if N < self.quantiles:
            x = np.array(sorted(t))
            y = np.arange(1.,float(N+1))/float(N)
        else:
            x = np.percentile(t,range(0,100+100//quantiles,100//quantiles))
            y = np.arange(0,1.+1./quantiles,1./quantiles)
        if self.reverse:
            y = y[::-1]
        if self.method == 'linear':
            self._warp = interp1d(x, y)
            self._inverse_warp = interp1d(y, x)
        else:
            x = x.reshape(-1,1)
            y = y.reshape(-1,1)
            self._warp = gpy.models.GPRegression(x,y,gpy.kern.RBF(1))
            self._warp.optimize()
        
    def warp(self,t):
        """ Warp from pseudotime to master time.
        
        :param t: pseudotime values
        :type t: Nx1 array of float
        :returns: inferred master time values
        :rtype: Nx1 array of float
        
        """
    
        if self.method == 'linear':
            return self._warp(t)
        else:
            return self._warp.predict(t.reshape(-1,1))[0]
        
    def inverse_warp(self,tm):
        """ Warp from master time to pseudotime .
    
        :param tm: master time values
        :type tm: Nx1 array of float
        :returns: inferred pseudotime values
        :rtype: Nx1 array of float
        
        """    
        if self.method == 'linear':
           return self._inverse_warp(np.clip(tm,0,1))
        else:
            return self._warp.infer_newX(tm.reshape(-1,1))[0]
        
    def plot(self):
        if self.method == 'linear':
            xnew = np.linspace(min(self._warp.x),max(self._warp.x))
            f = plt.figure()
            return plt.plot(self._warp.x,self._warp.y,'kx',xnew,self._warp(xnew),'b-')
        else:
            return self._warp.plot()

class MATCHER:
    """ Manifold Alignment to Characterize Experimental Relationships (MATCHER)
        
        :param X: One or more single cell datasets. To learn a joint trajectory from two datasets with known correspondences, add a second level list.
        :type X: list, possibly nested
        :param num_inducing: Number of "inducing inputs" to use in variational approximation
        :type missing_data: integer
        
    """
    def __init__(self,X,num_inducing=10):
        """ Constructor for MATCHER class """
        self.X = X
        self.model = []
        for i in range(len(self.X)):
            if isinstance(X[i],list):
                self.model.append(gpy.models.mrd.MRD(self.X[i], 1, num_inducing=num_inducing, kernel=gpy.kern.RBF(1)))
            else:
                 self.model.append(gpy.models.bayesian_gplvm.BayesianGPLVM(self.X[i], input_dim=1, kernel=gpy.kern.RBF(1)))
        self.warp_functions = []
        self.master_time = []

    def infer(self,quantiles=50,method=None,reverse=None):
        """ Infer pseudotime and master time values.
            
            :param quantiles: How many quantiles to use when computing warp functions
            :type quantiles: int
            :param method: Gaussian process regression or linear interpolation?
            :type method: list of strings
            :param reverse: Reverse pseudotime?
            :type quantiles: list of booleans
            
        """
        num_datasets = len(self.X)
        self.warp_functions = [None] * num_datasets
        self.master_time = [None] * num_datasets
        if method is None:
            method = ['gp']
            method = np.repeat(method,[num_datasets])
        if reverse is None:
            reverse = [False]
            reverse = np.repeat(reverse,[num_datasets])
        for i in range(num_datasets):
            self.model[i].optimize(messages=1)
            self.learn_warp_function(i,quantiles,method[i],reverse[i])
            
    def learn_warp_function(self,ind,quantiles=50,method='gp',reverse=False):
        t = self.model[ind].latent_space.mean
        self.warp_functions[ind] = WarpFunction(t,quantiles,method,reverse)
        self.master_time[ind] = self.warp_functions[ind].warp(t)
        
    def sample_master_time(self, ind, samples=10):
        """ Sample from the posterior for the inferred master time values for the specified model.
        
            :param ind: Index of model to sample
            :type ind: int
            :param samples: Number of samples
            :type samples: int
            :returns: Posterior samples from inferred master_time values for each cell
            :rtype: SxN array, where S is the number of samples and N is the number of cells
        
        """
        means = self.model[ind].latent_space.mean.flatten()
        variances = self.model[ind].latent_space.variance.flatten()
        sampled = np.random.multivariate_normal(means,np.diag(variances),samples)
        mapped = self.warp_functions[ind].warp(sampled.reshape(-1,1)).reshape(samples,-1)
        return mapped
            
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

    def infer_corresponding_features(self,model1,model2,inds1,inds2):
        """ Infer corresponding values of features from different data types
        
            :param model1: Index of first data type
            :type model1: int
            :param model2: Index of second data type
            :type model2: int
            :param inds1: Indices of features
            :type inds1: list
            :param inds2: Indices of features
            :type inds2: list
            :returns: Feature values in corresponding cells
            :rytpe: list of 2D arrays
            
        """
        t = self.master_time[model1]
        model2_internal = self.warp_functions[model2].inverse_warp(t.reshape(-1,1))
        vals1 = self.X[model1][:,inds1]
        vals2 = self.model[model2].predict(model2_internal)[0][:,inds2]
        return [vals1,vals2]

    def plot_corresponding_features(self,model1,model2,ind1,ind2):
        """ Infer corresponding values of features from different data types
        and displays a scatter plot of these values. Points are colored by
        master time.
        
            :param model1: Index of first data type
            :type model1: int
            :param model2: Index of second data type
            :type model2: int
            :param ind1: Index of feature
            :type ind1: int
            :param ind2: Index of feature
            :type ind2: int
            
        """
        t = self.master_time[model1]
        model2_internal = self.warp_functions[model2].inverse_warp(t.reshape(-1,1))
        vals1 = self.X[model1][:,ind1]
        vals2 = self.model[model2].predict(model2_internal)[0][:,ind2]
        plt.scatter(vals1,vals2,c=t,cmap='hot')
        cb = plt.colorbar()
        cb.set_label('Master Time')    

    def correlation(self,model1,model2,inds1,inds2,method="Spearman"):
        """ Approximate correlation between the specified features of two different data types.
        
            :param model1: Index of first data type
            :type model1: int
            :param model2: Index of second data type
            :type model2: int
            :param inds1: Indices of features
            :type inds1: list
            :param inds2: Indices of features
            :type inds2: list
            :param method: Type of correlation coefficient to compute ("Spearman" or "Pearson")
            :type method: string
            :returns: Correlation matrix
            :rytpe: 2D array
            
        """
        t = self.master_time[model1]
        model2_internal = self.warp_functions[model2].inverse_warp(t.reshape(-1,1))
        vals1 = self.X[model1][:,inds1]
        vals2 = self.model[model2].predict(model2_internal)[0][:,inds2]
        n1 = len(inds1)
        n2 = len(inds2)
        n = n1 + n2
        if method=="Spearman":
            corr_res = spearmanr(vals1,vals2)
            corr_res = corr_res[0]
        elif method == "Pearson":
            corr_res = np.corrcoef(vals1.transpose(),vals2.transpose())
        else:
            corr_res = np.corrcoef(vals1.transpose(),vals2.transpose())
        
        corr_mat = corr_res[range(n1),n1:n]
        return corr_mat
        
    def plot_warp_functions(self,inds):
        """ Plot the functions for each data type that map from domain-specific pseudotime to master time.
        
            :param inds: Indices of data types
            :type inds: list
        
        """
        fig, axes = plt.subplots(1,len(inds))
        for i in range(len(inds)):
            self.warp_functions[inds[i]].plot(ax=axes[i])
            t = self.model[inds[i]].latent_space.mean
            axes[i].set_xlim([min(t),max(t)])
            axes[i].set_ylim([0,1])
            axes[i].legend(loc='best')
            
    def plot_feature(self,model_ind,feature_ind):
        """ Plot the specified feature and its model fit
        
            :param model_ind: Index for data type
            :type model_ind: int
            :param feature_ind: Index of feature
            :type feature_ind: int
        
        """
        fig = self.model[model_ind].plot_f(which_data_ycols=[feature_ind],plot_inducing=False)
        fig.scatter(self.model[model_ind].latent_space.mean,self.X[model_ind][:,feature_ind],marker='x',c="black")
        
    def plot_master_time(self,inds):
        """ Plot inferred master time values and uncertainty for models specified by inds.
        
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
            ci_up = self.warp_functions[i].warp(ci_up.reshape(-1,1)).flatten()
            ci_down = means[axis_order] - 2 * np.sqrt(vars[axis_order])
            ci_down = self.warp_functions[i].warp(ci_down.reshape(-1,1)).flatten()
            means = self.warp_functions[i].warp(means.reshape(-1,1)).flatten()
            lines.extend(axes[i].plot(x, means[axis_order], c='b', label=r"$\mathbf{{T_{{{}}}}}$".format(i+1)))
            fills.append(axes[i].fill_between(x, ci_down, ci_up, facecolor=lines[i].get_color(), alpha=.3))
            axes[i].legend(borderaxespad=0.,loc=0)
            axes[i].set_xlim(min(x), max(x))
            axes[i].set_ylim(min(means), max(means))
        plt.draw()
        axes[len(self.model)-1].figure.tight_layout(h_pad=.01)
        return dict(lines=lines, fills=fills, bg_lines=bg_lines)
