import matplotlib.pyplot as plt 

from mpl_toolkits.mplot3d import Axes3D

import numpy as np

from typing import Optional

def plot_cell_state_2d(
    cell_state_array: np.ndarray,
    ax=None,
    connect: bool = False,  
    fig_args: dict = None, 
    title: Optional[str] = None,
    **kwargs
) -> plt.axes:
    """Plot a 2D representation of a cell_states received from dimensionality reduction.  

    Parameters
    ----------
    cell_state_array : np.ndarray
        _description_
    ax : _type_, optional
        _description_, by default None
    title : Optional[str], optional
        _description_, by default None

    Returns
    -------
    plt.axes
        _description_
    """
    # If the plot is not used as a subplot, and gets an ax passed, create one.  
    if ax is None: 
        fig, ax = plt.subplots(**fig_args)

    if connect: 
        ax.plot(cell_state_array[:,0], cell_state_array[:,1], **kwargs)
    else:
        ax.scatter(cell_state_array[:,0], cell_state_array[:,1], **kwargs)
    
    if title is not None: 
        ax.set_title(title)

    return ax


def plot_cell_state_3d(
    cell_state_array: np.ndarray,
    ax=None,
    connect: bool = False,
    fig_args: dict = None,
    ax_view_params: dict = {"elev": 10, "azim": 45}, 
    title: Optional[str] = None,
    **kwargs
) -> plt.axes:
    """Plot a 3D representation of a cell_states received from dimensionality reduction.  

    Parameters
    ----------
    cell_state_array : np.ndarray
        _description_
    ax : _type_, optional
        _description_, by default None
    title : Optional[str], optional
        _description_, by default None

    Returns
    -------
    plt.axes
        _description_
    """
    # If the plot is not used as a subplot, and gets an ax passed, create one.  
    if ax is None: 
        fig = plt.figure(**fig_args)
        ax = fig.add_subplot(111, projection="3d")

    if connect: 
        ax.plot(cell_state_array[:,0], cell_state_array[:,1], cell_state_array[:,2], **kwargs)
    else:
        ax.scatter(cell_state_array[:,0], cell_state_array[:,1], cell_state_array[:,2], **kwargs)
    
    ax.view_init(**ax_view_params)
    

    if title is not None: 
        ax.set_title(title)

    return ax
