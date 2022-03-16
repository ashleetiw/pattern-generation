import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
import seaborn as sns

tfd = tfp.distributions
tfb = tfp.bijectors
tfk = tfp.math.psd_kernels
sns.set_style('whitegrid')
np.random.seed(42)

class kernel:
    def __init__(self):
        self.x=2

    def plot_kernel(self,X, y, cov, description, fig, subplot_spec, xlim,scatter=False, rotate_x_labels=False):
        """Plot kernel matrix and samples."""
        grid_spec = gridspec.GridSpecFromSubplotSpec(1, 2, width_ratios=[2,1], height_ratios=[1],
            wspace=0.18, hspace=0.0,subplot_spec=subplot_spec)
        ax1 = fig.add_subplot(grid_spec[0])
        ax2 = fig.add_subplot(grid_spec[1])
        # Plot samples
        if scatter:
            for i in range(y.shape[1]):
                ax1.scatter(X, y[:,i], alpha=0.8, s=3)
        else:
            for i in range(y.shape[1]):
                ax1.plot(X, y[:,i], alpha=0.8)
        ax1.set_ylabel('$y$', fontsize=13, labelpad=0)
        ax1.set_xlabel('$x$', fontsize=13, labelpad=0)
        ax1.set_xlim(xlim)
        if rotate_x_labels:
            for l in ax1.get_xticklabels():
                l.set_rotation(30)
        ax1.set_title(f'Samples from {description}')
        # Plot covariance matrix

        im = ax2.imshow(cov, cmap=cm.YlOrRd)
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='5%', pad=0.02)
        cbar = plt.colorbar(im, ax=ax2, cax=cax)
        cbar.ax.set_ylabel('$K(X,X)$', fontsize=8)
        ax2.set_title(f'Covariance matrix\n{description}')
        ax2.set_xlabel('X', fontsize=10, labelpad=0)
        ax2.set_ylabel('X', fontsize=10, labelpad=0)
        # Show 5 custom ticks on x an y axis of covariance plot
        nb_ticks = 5
        ticks = list(range(xlim[0], xlim[1]+1))
        ticks_idx = np.rint(np.linspace(
            1, len(ticks), num=min(nb_ticks,len(ticks)))-1).astype(int)
        ticks = list(np.array(ticks)[ticks_idx])
        ax2.set_xticks(np.linspace(0, len(X), len(ticks)))
        ax2.set_yticks(np.linspace(0, len(X), len(ticks)))
        ax2.set_xticklabels(ticks)
        ax2.set_yticklabels(ticks)
        if rotate_x_labels:
            for l in ax2.get_xticklabels():
                l.set_rotation(30)
        ax2.grid(False)

    def ExponentiatedQuadratic(self,nb_of_samples, nb_of_realizations):
        # Generate input points
        xlim = (-4, 4)
        X = np.expand_dims(np.linspace(*xlim, nb_of_samples), 1)
        # Start plotting
        fig = plt.figure(figsize=(7, 10)) 
        gs = gridspec.GridSpec(4, 1, figure=fig, wspace=0.2, hspace=0.4)

        # Plot first
        cov = tfk.ExponentiatedQuadratic(amplitude=1., length_scale=1.).matrix(X, X).numpy()
        
        y = np.random.multivariate_normal(
            mean=np.zeros(nb_of_samples), cov=cov, 
            size=nb_of_realizations).T

        self.plot_kernel(
            X, y,cov, '$\\ell = 1$, $\\sigma = 1$', 
            fig, gs[0], xlim)

        # Plot second
        cov = tfk.ExponentiatedQuadratic(amplitude=1., length_scale=0.3).matrix(X, X).numpy()
        y = np.random.multivariate_normal(
            mean=np.zeros(nb_of_samples), cov=cov, 
            size=nb_of_realizations).T
        self.plot_kernel(
            X, y, cov, '$\\ell = 0.3$, $\\sigma = 1$', 
            fig, gs[1], xlim)

        # Plot second
        cov = tfk.ExponentiatedQuadratic(amplitude=1., length_scale=2.).matrix(X, X).numpy()
        y = np.random.multivariate_normal(
            mean=np.zeros(nb_of_samples), cov=cov, 
            size=nb_of_realizations).T
        self.plot_kernel(
            X, y, cov, '$\\ell = 2$, $\\sigma = 1$', 
            fig, gs[2], xlim)

        # Plot third
        cov = tfk.ExponentiatedQuadratic(amplitude=10., length_scale=1.).matrix(X, X).numpy()
        y = np.random.multivariate_normal(
            mean=np.zeros(nb_of_samples), cov=cov, 
            size=nb_of_realizations).T
        self.plot_kernel(
            X, y, cov, '$\\ell = 1$, $\\sigma = 10$',  
            fig, gs[3], xlim)

        plt.suptitle('Exponentiated quadratic', y=0.99)
        fig.subplots_adjust(
            left=0.07, bottom=0.04, right=0.93, top=0.94)
        plt.savefig('f1.png')
        

    def RationalQuadratic(self,nb_of_samples, nb_of_realizations):
        xlim = (-5, 5)
        X = np.expand_dims(np.linspace(*xlim, nb_of_samples), 1)

        # Start plotting
        fig = plt.figure(figsize=(7, 12))
        gs = gridspec.GridSpec(
            5, 1, figure=fig, wspace=0.2, hspace=0.5)

        # Plot first
        cov = tfk.RationalQuadratic(amplitude=1., length_scale=1., scale_mixture_rate=1.).matrix(X, X).numpy()
        y = np.random.multivariate_normal(
            mean=np.zeros(nb_of_samples), cov=cov, 
            size=nb_of_realizations).T
        self.plot_kernel(
            X, y,cov, '$\\ell=1$, $\\alpha=1$', 
            fig, gs[0], xlim)

        # Plot second
        cov = tfk.RationalQuadratic(amplitude=1., length_scale=5., scale_mixture_rate=1.).matrix(X, X).numpy()
        y = np.random.multivariate_normal(
            mean=np.zeros(nb_of_samples), cov=cov, 
            size=nb_of_realizations).T
        self.plot_kernel(
            X, y,cov, '$\\ell=5$, $\\alpha=1$', 
            fig, gs[1], xlim)

        # Plot third
        cov = tfk.RationalQuadratic(amplitude=1., length_scale=0.2, scale_mixture_rate=1.).matrix(X, X).numpy()
        y = np.random.multivariate_normal(
            mean=np.zeros(nb_of_samples), cov=cov, 
            size=nb_of_realizations).T
        self.plot_kernel(
            X, y,cov, '$\\ell=0.2$, $\\alpha=1$', 
            fig, gs[2], xlim)

        # Plot fourth
        cov = tfk.RationalQuadratic(amplitude=1., length_scale=1., scale_mixture_rate=0.1).matrix(X, X).numpy()
        y = np.random.multivariate_normal(
            mean=np.zeros(nb_of_samples), cov=cov, 
            size=nb_of_realizations).T
        self.plot_kernel(
            X, y,cov, '$\\ell=1$, $\\alpha=0.1$', 
            fig, gs[3], xlim)

        # Plot fifth
        cov = tfk.RationalQuadratic(amplitude=1., length_scale=1., scale_mixture_rate=1000.).matrix(X, X).numpy()
        y = np.random.multivariate_normal(
            mean=np.zeros(nb_of_samples), cov=cov, 
            size=nb_of_realizations).T
        self.plot_kernel(
            X, y,cov, '$\\ell=1$, $\\alpha=1000$', 
            fig, gs[4], xlim)

        fig.suptitle('Rational quadratic ($\\sigma=1$)', y=0.99)
        fig.subplots_adjust(
            left=0.06, bottom=0.04, right=0.94, top=0.95)
        plt.savefig('f2.png')


    def periodic_tf(self,length_scale, period):
        """Periodic kernel TensorFlow operation."""
        amplitude_tf = tf.constant(1, dtype=tf.float64)
        length_scale_tf = tf.constant(length_scale, dtype=tf.float64)
        period_tf = tf.constant(period, dtype=tf.float64)
        kernel = tfk.ExpSinSquared(
            amplitude=amplitude_tf, 
            length_scale=length_scale_tf,
            period=period_tf)
        return kernel

    def periodic(self,xa, xb, length_scale, period):
        """Evaluate periodic kernel."""
        kernel = self.periodic_tf(length_scale, period)
        kernel_matrix = kernel.matrix(xa, xb)
        with tf.Session() as sess:
            return sess.run(kernel_matrix)

    def ExpSinSquared(self,nb_of_samples, nb_of_realizations):

        xlim = (-2, 2)
        X = np.expand_dims(np.linspace(*xlim, nb_of_samples), 1)
        # Start plotting
        fig = plt.figure(figsize=(7, 12))
        gs = gridspec.GridSpec(5, 1, figure=fig, wspace=0.2, hspace=0.5)

        # Plot first
        cov = tfk.ExpSinSquared(amplitude=1., length_scale=1., period=1.).matrix(X, X).numpy()
        y = np.random.multivariate_normal(
            mean=np.zeros(nb_of_samples), cov=cov, 
            size=nb_of_realizations).T
        self.plot_kernel(X, y, cov, '$\\ell=1$, $p=1$', 
            fig, gs[0], xlim)

        # Plot second
        cov = tfk.ExpSinSquared(amplitude=1., length_scale=2., period=1.).matrix(X, X).numpy()
        y = np.random.multivariate_normal(
            mean=np.zeros(nb_of_samples), cov=cov, 
            size=nb_of_realizations).T
        self.plot_kernel(
            X, y, cov, '$\\ell=2$, $p=1$', 
            fig, gs[1], xlim)

        # Plot third
        cov = tfk.ExpSinSquared(amplitude=1., length_scale=0.5, period=1.).matrix(X, X).numpy()
        y = np.random.multivariate_normal(
            mean=np.zeros(nb_of_samples), cov=cov, 
            size=nb_of_realizations).T
        self.plot_kernel(
            X, y, cov, '$\\ell=0.5$, $p=1$', 
            fig, gs[2], xlim)

        # Plot fourth
        cov = tfk.ExpSinSquared(amplitude=1., length_scale=1., period=0.5).matrix(X, X).numpy()
        y = np.random.multivariate_normal(
            mean=np.zeros(nb_of_samples), cov=cov, 
            size=nb_of_realizations).T
        self.plot_kernel(
            X, y, cov, '$\\ell=1$, $p=0.5$', 
            fig, gs[3], xlim)

        # Plot fifth
        cov = tfk.ExpSinSquared(amplitude=1., length_scale=1., period=2.).matrix(X, X).numpy()
        y = np.random.multivariate_normal(
            mean=np.zeros(nb_of_samples), cov=cov, 
            size=nb_of_realizations).T
        self.plot_kernel(
            X, y, cov, '$\\ell=1$, $p=2$',
            fig, gs[4], xlim)

        fig.suptitle('Periodic ($\\sigma=1$)', y=0.99)
        fig.subplots_adjust(
            left=0.06, bottom=0.04, right=0.94, top=0.95)
        plt.savefig('f3.png')


    def get_local_periodic_kernel(self,periodic_length_scale, period, amplitude, local_length_scale):
        periodic = tfk.ExpSinSquared(amplitude=amplitude, length_scale=periodic_length_scale, period=period)
        local = tfk.ExponentiatedQuadratic(length_scale=local_length_scale)
        return periodic * local
    
    def local_periodic_kernel(self,nb_of_samples, nb_of_realizations):
        xlim = (-3, 3)
        X = np.expand_dims(np.linspace(*xlim, nb_of_samples), 1)

        # Start plotting
        fig = plt.figure(figsize=(7, 8))
        gs = gridspec.GridSpec(
            3, 1, figure=fig, wspace=0.2, hspace=0.4)

        # Plot first
        cov = self.get_local_periodic_kernel(periodic_length_scale=1., period=1., amplitude=1., local_length_scale=1.).matrix(X, X).numpy()
        y = np.random.multivariate_normal(
            mean=np.zeros(nb_of_samples), cov=cov, 
            size=nb_of_realizations).T
        self.plot_kernel(
            X, y, cov, '$\\ell_{eq}=1$', 
            fig, gs[0], xlim)

        # Plot second
        cov = self.get_local_periodic_kernel(periodic_length_scale=1., period=1., amplitude=1., local_length_scale=0.5).matrix(X, X).numpy()
        y = np.random.multivariate_normal(
            mean=np.zeros(nb_of_samples), cov=cov, 
            size=nb_of_realizations).T
        self.plot_kernel(
            X, y, cov, '$\\ell_{eq}=0.5$', 
            fig, gs[1], xlim)

        # Plot third
        cov = self.get_local_periodic_kernel(periodic_length_scale=1., period=1., amplitude=1., local_length_scale=3.).matrix(X, X).numpy()
        y = np.random.multivariate_normal(
            mean=np.zeros(nb_of_samples), cov=cov, 
            size=nb_of_realizations).T
        self.plot_kernel(
            X, y, cov, '$\\ell_{eq}=3$', 
            fig, gs[2], xlim)

        fig.suptitle('Local periodic ($\\sigma=1$, $\\ell_p=1$, $p=1$)', y=0.99)
        fig.subplots_adjust(left=0.07, bottom=0.05, right=0.93, top=0.91)
        plt.savefig('f4.png')
        
k=kernel()
k.ExponentiatedQuadratic(150,3)
# k.RationalQuadratic(150,3)
# k.ExpSinSquared(150,3)
# k.local_periodic_kernel(150,3)
plt.show()