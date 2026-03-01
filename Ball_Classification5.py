from sklearn import tree

# Feature encoding
# Rough : 1
# Smooth : 0

# Label encoding
# tennis : 1
# cricket : 2

def main():
    print("Ball Classification Case Study")

    # Original encoded data set
    # Independent variables
    X = [[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[96,0],[43,1],[110,0],[35,1],[95,0]] 

    # Dependent variables
    Y = [1,1,2,1,2,1,2,1,1,1,2,1,2,1,2]

    #-------------------------------------------------------------------
    # Independent variables for training
    Xtrain = [[35,1],[47,1],[90,0],[48,1],[90,0],[35,1],[92,0],[35,1],[35,1],[35,1],[96,0],[43,1],[110,0]] 

    # Independent variables for test
    Xtest = [[35,1],[95,0]] 

    # Dependent variables for training
    Ytrain = [1,1,2,1,2,1,2,1,1,1,2,1,2]

    # Dependent variables for test
    Ytest = [1,2]

    # ------------------------------------------------------------------

    modelobj = tree.DecisionTreeClassifier()

    # model training
    trainedmodel = modelobj.fit(Xtrain,Ytrain)

    Result = trainedmodel.predict(Xtest)

    print("Model predicts the object as",Result)        # 1 and 2

if __name__ == "__main__":
    main()

