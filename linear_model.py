import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

"""Functions used in the algorithm"""
def convert_theta(theta,scaler):
    if scaler!=None:
        M=scaler.mean_
        s=scaler.scale_
        thetaprim=theta/s
    else:
        thetaprim=theta
    return thetaprim

def relu(x):
    result=np.maximum(x, 0)
    return result

def heaviside(x):
    result=np.where(x<0,0,1)
    return result

def methode(method,x):
    if method=="Exponential":
        f1=np.exp(x)
        f2=np.exp(x)

    elif method=="Squaredrelu":
        f1=relu(x)**2
        f2=heaviside(x)*relu(x)

    elif method=="Relu":
        f1=relu(x)
        f2=heaviside(x)
    return f1,f2

class constrained_linear_model:
    """
    A linear regression model that allows to set two types of linear constraints:
    -Constraints on Theta1 & Theta2 parameters: CSTR1=Theta1-p1*Theta2>0
    -Bounds on a parameter Theta: CSTR2=p2*(Theta-bound)>0 where p2=1 or p2=-1 depending on the inequality
    These constraints are implemented by adding to the loss function a convex penalty function of that type: P=Sum(f(-CSTR)),
    where f can be a Relu, Squared Relu or an Exponential function.

    :init: Initialization of the weights. For random init, set to "rand"
    :scaled: Normalization of the features. For standardization: set to True
    :fit_intercept: When features are standardized, must be set to True
    :alpha: Learning rate
    :num_iteration: Number of training iterations
    :reg: Amplitude of the regularization
    :k: Amplitude of the constraints
    :cst_dict1: List of constraints of that type [Theta1_index,Theta2_index,p1]
    :cst_dict2: List of constraints of that type [Theta1_index,bound,p2]
    :method: Function f implemented in the penalty function
    :verbose: Set to True, display logs when training
    """
    def __init__(self,k,reg,alpha,num_iteration,fit_intercept,method,init,scaled,cst_dict1,cst_dict2, verbose):
        
        #Input
        self.num_iteration=num_iteration
        self.fit_intercept=fit_intercept
        self.k=k
        self.reg=reg
        self.alpha=alpha
        self.method=method 
        self.init=init
        self.scaled=scaled
        self.cst_dict1=cst_dict1
        self.cst_dict2=cst_dict2
        self.verbose=verbose
        
        #Output
        self.coef=None
        self.coef_full=None
        self.COST1_liste=None
        self.COST2_liste=None
        self.COST_liste=None
        self.ypred=None
        self.cols=None
        self.drop_cols=None
        self.scaler=None
        
    def fit(self,X,y):
        
        num_iteration=self.num_iteration
        fit_intercept=self.fit_intercept
        k=self.k
        reg=self.reg
        alpha=self.alpha
        method=self.method 
        cst_dict1=self.cst_dict1
        cst_dict2=self.cst_dict2
        init=self.init
        scaled=self.scaled
        verbose=self.verbose
        self.X=X
        m=len(X)
        
        #Cleaning unobserved features to avoid collinearity
        cols=X.columns
        cols_f=X[X>0].dropna(axis=1,how='all').columns   
        drop_cols=[c for c in cols if c not in cols_f]
        self.cols=cols
        self.drop_cols=drop_cols
        X=X[cols_f]
        n=len(X.columns)
        
        #Standardisation of the features
        if scaled ==True:
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X))
        else:
            scaler=None
        self.scaler=scaler

        #Initialization of the weights
        np.random.seed(1)
        b=np.random.randn(1)/100
        if init=="rand":
            theta0=np.random.randn(n)/100            
        else:
             theta0=np.zeros(n)
        theta=theta0  
        
        #Loss function
        COST_liste=[]
        COST1_liste=[]
        COST2_liste=[]

        #Fitting intercept
        if fit_intercept==True:
            z=1
        else:
            z=0

        #Batch gradient descent with a penalty function
        for i in range(num_iteration):

            #Compute error          
            ypred=(X.dot(theta)).tolist()+b*z
            self.ypred=ypred
            Error=ypred-y.to_numpy()
            
            #Build the constraints on the original weights
            theta_conv=convert_theta(theta,self.scaler)
            
            contraintes1=constraint(theta_conv,cst_dict1,1,method,n)
            contraintes1.compute_mat()
            mat01=contraintes1.cst_mat0
            mat11=contraintes1.cst_mat1
            mat21=contraintes1.cst_mat2
            taux1=((mat01<0).sum())/len(cst_dict1)*100

            contraintes2=constraint(theta_conv,cst_dict2,2,method,n)
            contraintes2.compute_mat()
            mat02=contraintes2.cst_mat0
            mat12=contraintes2.cst_mat1
            mat22=contraintes2.cst_mat2
            taux2=((mat02<0).sum())/len(cst_dict2)*100
            
            #Compute the loss function
            COST1=1/(2*m)*((Error)**2).sum()+reg/2*(theta**2).sum()            
            COST2=(k*mat11+k*mat12).sum()
            COST=COST1+COST2

            COST_liste.append(COST)
            COST1_liste.append(COST1)
            COST2_liste.append(COST2)
            
            self.COST1_liste=COST1_liste
            self.COST2_liste=COST2_liste
            self.COST_liste=COST_liste
            
            #Training logs
            if verbose==True and i%10==0:
                print("Epoch "+str(i),"\n",
                      "COST:",round(COST,8),round(COST1,8),round(COST2,8),"\n",
                      "theta:", np.round(np.array(theta_conv)[:5],10),"\n",
                      "TauxCSS1",round(taux1,5),"%","\n",
                      "TauxCSS2",round(taux2,5),"%","\n")

            #Update the weights after computing the partial derivatives
            K0=-alpha/m*np.array(Error.dot(X))
            K1=-alpha*(k*mat21+k*mat22)
            K2=-alpha*reg*theta.T.flatten()

            theta=theta+K0+K1+K2
            b=b-alpha/m*Error.sum()

        #Return the original weights
        self.coef=convert_theta(theta,self.scaler)
        self.coef_full=self._fill_coef()

    #Compute final weights list
    def _fill_coef(self):
        cols=self.cols
        drop_cols=self.drop_cols
        coef=self.coef
        coef_full=[]
        i=0
        for c in cols:
            if c in drop_cols:
                coef_full.append(None)
            else:
                coef_full.append(coef[i]) 
                i+=1
        return coef_full
    
    #Display the loss with regards to each iteration
    def plot_cost(self,roll):
        COST1_liste=self.COST1_liste
        COST2_liste=self.COST2_liste
        fig, axe=plt.subplots(1,1,figsize=(20,5), sharey=True) 
        plt.scatter(list(range(len(COST2_liste))), COST2_liste,color="blue",label="CONTRAINTE")
        plt.scatter(list(range(len(COST1_liste))), COST1_liste,color="green",label="MSE")
        plt.legend()

    def plot_pred(self,y):
        ypred=self.ypred
        plt.scatter(y,ypred)

class constraint:
    """
    A class that build constraints array both for penalty function
    and partial derivatives computations
    
    :theta: List of weights
    :cst_dict: List of constraints of the constrainedBGD
    :method: method chosen for the constrainedBGD
    :n: lenght of the training set
    """
    
    def __init__(self,theta,cst_dict,kind,method,n):
        
        self.theta=theta
        self.kind=kind
        self.method=method
        self.cst_dict=cst_dict
        self.n=n
        self.cst_mat0=None
        self.cst_mat1=None
        self.cst_mat2=None
        
    def compute_mat(self):
        theta=self.theta
        kind=self.kind
        cst_dict=self.cst_dict
        method=self.method
        n=self.n

        cst_mat0=np.zeros(n)
        cst_mat1=np.zeros(n)
        cst_mat2=np.zeros(n)

        if kind==1:
            for cs in cst_dict:
                #Array with penalty terms
                cst_mat0[cs[0]]+=-(theta[cs[0]]-cs[2]*theta[cs[1]])
                cst_mat1[cs[0]]+=methode(method,-(theta[cs[0]]-cs[2]*theta[cs[1]]))[0] 
                #Array with derivatives
                cst_mat2[cs[0]]+=-methode(method,-(theta[cs[0]]-cs[2]*theta[cs[1]]))[1]
                cst_mat2[cs[1]]+=cs[2]*methode(method,-(theta[cs[0]]-cs[2]*theta[cs[1]]))[1] 

        elif kind==2:
            for cs in cst_dict:
                #Array with penalty terms
                cst_mat0[cs[0]]+=-cs[2]*(theta[cs[0]]-cs[1])
                cst_mat1[cs[0]]+=methode(method,-cs[2]*(theta[cs[0]]-cs[1]))[0] 
                #Array with derivatives
                cst_mat2[cs[0]]+=-cs[2]*methode(method,-cs[2]*(theta[cs[0]]-cs[1]))[1]
       
        self.cst_mat0=cst_mat0
        self.cst_mat1=cst_mat1
        self.cst_mat2=cst_mat2





            
