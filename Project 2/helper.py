def visualize_decision_boundary(predict, 
                                 X, y, 
                                 attribute_names,
                                 class_names,
                                 train=None, test=None, 
                                 delta=5e-3,
                                 show_legend=True):
    '''
    Visualize the decision boundary of a classifier trained on a 2 dimensional
    input feature space.
    
    Creates a grid of points based on ranges of features in X, then determines
    classifier output for each point. The predictions are color-coded and plotted
    along with the data and a visualization of the partitioning in training and
    test if provided.
    
    Args:
        predict:
                A lambda function that takes the a grid of shape [M, N] as 
                input and returns the prediction of the classifier. M corre-
                sponds to the number of features (M==2 required), and N corre-
                sponding to the number of points in the grid. Can e.g. be a 
                trained PyTorch network (torch.nn.Sequential()), such as trained
                using toolbox_02450.train_neural_network, where the provided
                function would be something similar to: 
                >>> predict = lambda x: (net(torch.tensor(x, dtype=torch.float))).data.numpy()
                
        X:      A numpy array of shape (N, M), where N is the number of 
                observations and M is the number of input features (constrained
                to M==2 for this visualization).
                If X is a list of len(X)==2, then each element in X is inter-
                preted as a partition of training or test data, such that 
                X[0] is the training set and X[1] is the test set.
                
        y:      A numpy array of shape (N, 1), where N is the number of 
                observations. Each element is either 0 or 1, as the 
                visualization is constrained to a binary classification
                problem.
                If y is a list of len(y)==2, then each element in y is inter-
                preted as a partion of training or test data, such that 
                y[0] is the training set and y[1] is the test set. 
                
        attribute_names:
                A list of strings of length 2 giving the name
                of each of the M attributes in X.
                
        class_names: 
                A list of strings giving the name of each class in y.
                
        train (optional):  
                A list of indices describing the indices in X and y used for
                training the network. E.g. from the output of:
                    sklearn.model_selection.KFold(2).split(X, y)
                    
        test (optional):   
                A list of indices describing the indices in X and y used for
                testing the network (see also argument "train").
                
        delta (optional):
                A float describing the resolution of the decision
                boundary (default: 0.01). Default results grid of 100x100 that
                covers the first and second dimension range plus an additional
                25 percent.
        show_legend (optional):
                A boolean designating whether to display a legend. Defaults
                to True.
                
    Returns:
        Plots the decision boundary on a matplotlib.pyplot figure.
        
    '''
    
    import torch
    
    C = len(class_names)
    if isinstance(X, list) or isinstance(y, list):
        assert isinstance(y, list), 'If X is provided as list, y must be, too.'
        assert isinstance(y, list), 'If y is provided as list, X must be, too.'
        assert len(X)==2, 'If X is provided as a list, the length must be 2.'
        assert len(y)==2, 'If y is provided as a list, the length must be 2.'
        
        N_train, M = X[0].shape
        N_test, M = X[1].shape
        N = N_train+N_test
        grid_range = get_data_ranges(np.concatenate(X))
    else:
        N, M = X.shape
        grid_range = get_data_ranges(X)
    assert M==2, 'TwoFeatureError: Current neural_net_decision_boundary is only implemented for 2 features.'
    # Convert test/train indices to boolean index if provided:
    if train is not None or test is not None:
        assert not isinstance(X, list), 'Cannot provide indices of test and train partition, if X is provided as list of train and test partition.'
        assert not isinstance(y, list), 'Cannot provide indices of test and train partition, if y is provided as list of train and test partition.'
        assert train is not None, 'If test is provided, then train must also be provided.'
        assert test is not None, 'If train is provided, then test must also be provided.'
        train_index = np.array([(int(e) in train) for e in np.linspace(0, N-1, N)])
        test_index = np.array([(int(e) in test) for e in np.linspace(0, N-1, N)])
    
    xx = np.arange(grid_range[0], grid_range[1], delta)
    yy = np.arange(grid_range[2], grid_range[3], delta)
    # make a mesh-grid from a and b that spans the grid-range defined
    grid = np.stack(np.meshgrid(xx, yy))
    # reshape grid to be of shape "[number of feature dimensions] by [number of points in grid]"
    # this ensures that the shape fits the way the network expects input to be shaped
    # and determine estimated class label for entire featurespace by estimating
    # the label of each point in the previosly defined grid using provided
    # function predict()
    grid_predictions = predict(np.reshape(grid, (2,-1)).T)
    
    # Plot data with color designating class and transparency+shape
    # identifying partition (test/train)
    if C == 2:
        c = ['r','b']
        cmap = cm.bwr
        vmax=1
    else:
        c = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
             'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
        cmap = cm.tab10
        vmax=10
        
    s = ['o','x']; t = [.33, 1.0];
    for i in range(C):
        if train is not None and test is not None:
            for j, e in enumerate([train_index, test_index]):
                idx = (np.squeeze(y)==i) & e
                plt.plot(X[idx, 0], X[idx, 1], s[j],color=c[i], alpha=t[j])
        if isinstance(X,list) and isinstance(y, list):
            for j, (X_par, y_par) in enumerate(zip(X,y)):
                idx = np.squeeze(y_par)==i
                h = plt.plot(X_par[idx, 0], X_par[idx, 1],s[j], color=c[i], alpha=t[j])
  
    plt.xlim(grid_range[0:2])
    plt.ylim(grid_range[2:])
    plt.xlabel(attribute_names[0]);
    plt.ylabel(attribute_names[1])

    # reshape the predictions for each point in the grid to be shaped like
    # an image that corresponds to the feature-scace using the ranges that
    # defined the grid (a and b)
    decision_boundary = np.reshape(grid_predictions, (len(yy), len(xx)))
    # display the decision boundary
    ax = plt.imshow(decision_boundary, cmap=cmap, 
           extent=grid_range, vmin=0, vmax=vmax, alpha=.33, origin='lower')
    plt.axis('auto')
    if C == 2:
        plt.contour(grid[0], grid[1], decision_boundary, levels=[.5])
        plt.colorbar(ax, fraction=0.046, pad=0.04);
    if show_legend:
        plt.legend([class_names[i]+' '+e for i in range(C) for e in ['train','test']],
                   bbox_to_anchor=(1.2,1.0))