import numpy as np
import numpy.typing as npt

class Data():
    r"""Data node for computational graphs with selective inference support.
    
    The Data class serves as a node in a computational graph that can either
    hold data directly or compute it from a parent node. It supports 
    parametrization for selective inference and provides methods for 
    data updates and inference computations.
    
    The inference computation follows the linear relationship:
    
    .. math::
        \mathbf{y} = \mathbf{a} + \mathbf{b} \cdot z
    
    where :math:`\mathbf{a}` and :math:`\mathbf{b}` are parametrization vectors
    and :math:`z` is the inference parameter.
    
    Parameters
    ----------
    parent : any, optional
        Parent node to compute data from. If None, data must be set directly.
        
    Attributes
    ----------
    data : array-like or None
        The actual data stored in this node
    parent : any or None
        Reference to parent node for computation
    a : array-like or None
        Linear intercept parameter for inference
    b : array-like or None  
        Linear coefficient parameter for inference
    inference_data : array-like or None
        Data used for inference computations
        
    Notes
    -----
    If parent is provided, data will be computed from parent when called.
    Otherwise, data must be set directly using the :py:meth:`update` method.
    
    Examples
    --------
    >>> data_node = Data()
    >>> data_node.update(np.array([[1, 2], [3, 4]]))
    >>> result = data_node()
    """
    def __init__(self, parent: any = None):
        self.data = None
        self.parent = parent
        self.a = None
        self.b = None
        self.inference_data = None
        
    def __call__(self) -> npt.NDArray[np.floating]:
        r"""Retrieve or compute the data from this node.
        
        If the node has a parent, it will compute the data from the parent.
        Otherwise, it returns the directly stored data.
        
        Returns
        -------
        data : array-like, shape (n, d)
            The data array from this node
            
        Raises
        ------
        ValueError
            If no data is available and no parent to compute from
            
        Examples
        --------
        >>> data_node = Data()
        >>> data_node.update(np.array([1, 2, 3]))
        >>> result = data_node()
        >>> print(result)
        [1 2 3]
        """
        if self.parent is None:
            if self.data is None:
                raise ValueError("Data node has no data or parent to compute from.")
            return self.data
        self.parent()
        return self.data

    def update(self, data: npt.NDArray[np.floating]):
        r"""Update the data stored in this node.
        
        Parameters
        ----------
        data : array-like, shape (n, d)
            New data to store in the node
            
        Examples
        --------
        >>> data_node = Data()
        >>> new_data = np.array([[1, 2], [3, 4]])
        >>> data_node.update(new_data)
        """
        self.data = data
        
    def parametrize(self, a: npt.NDArray[np.floating] = None, b: npt.NDArray[np.floating] = None, data: npt.NDArray[np.floating] = None):
        r"""Set parameters for selective inference computations.
        
        This method sets linear parameters that define the relationship
        between the data and inference variables in the form:
        
        .. math::
            \mathbf{y} = \mathbf{a} + \mathbf{b} \cdot z
        
        where :math:`\mathbf{a}` is the intercept, :math:`\mathbf{b}` is the 
        coefficient vector, and :math:`z` is the inference parameter.
        
        Parameters
        ----------
        a : array-like, shape (d,), optional
            Linear intercept parameter
        b : array-like, shape (d,), optional
            Linear coefficient parameter
        data : array-like, shape (n, d), optional
            Inference data to store
            
        Examples
        --------
        >>> data_node = Data()
        >>> a = np.array([1, 2])
        >>> b = np.array([0.5, 1.0])
        >>> data_node.parametrize(a=a, b=b)
        """
        self.a = a
        self.b = b
        self.inference_data = data

    def inference(self, z: float):
        r"""Perform inference computation with given parameter z.
        
        Computes the linear relationship :math:`\mathbf{y} = \mathbf{a} + \mathbf{b} \cdot z`
        and retrieves the inference interval from parent if available.
        
        Parameters
        ----------
        z : float
            Parameter value for inference computation
            
        Returns
        -------
        inference_data : array-like, shape (d,)
            Computed inference data :math:`\mathbf{a} + \mathbf{b} \cdot z`
        a : array-like, shape (d,)
            Linear intercept parameter
        b : array-like, shape (d,)
            Linear coefficient parameter
        interval : list of float
            Inference interval bounds :math:`[z_{min}, z_{max}]`
            
        Examples
        --------
        >>> data_node = Data()
        >>> data_node.parametrize(a=np.array([1]), b=np.array([2]))
        >>> result, a, b, interval = data_node.inference(0.5)
        >>> print(result)  # Should be [2.0] (1 + 2 * 0.5)
        [2.]
        """
        if self.parent is not None:
            interval = self.parent.inference(z)
        else:
            interval = [-np.inf, np.inf]
            
        if self.a is not None and self.b is not None:
            self.inference_data = self.a + self.b * z
            return self.inference_data, self.a, self.b, interval
        return self.inference_data, self.a, self.b, interval