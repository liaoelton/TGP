import random
import copy 
import math
import numpy as np
def addOne(x):
    return x+1

def subOne(x):
    return x-1

def log(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(x > 0.001, np.log(x), 0.)

def exp(x):
    try:
        res = np.exp(x)
    except OverflowError:
        res = float('inf')
    return res

def sqrt(x):
    if x >= 0:
        return np.sqrt(x)
    else:
        return np.sqrt(-x)
def pow2(x):
    return x*x

def add(x,y):
    return x+y
def sub(x,y):
    return x-y
def mul(x,y):
    return x*y
def div(x,y):
    #print(x,y)
    #print(np.where(np.abs(y) > 0.001, np.divide(x, y), 1.))
    return np.where(np.abs(y) > 0.001, np.divide(x, y), 1.)

class Function:
    def __init__(self,func,name , arity):
        self.func = func
        self.name = name
        self.arity = arity
    def __call__(self,*x):
        return self.func(*x)

addOne_ = Function(addOne ,"addOne" ,1)
subOne_ = Function(subOne , "subOne",1)
log_ = Function(log , "log",1)
exp_ = Function(exp , "exp",1)
sqrt_ = Function(sqrt , "sqrt",1)
pow2_ = Function(pow2 , "pow2",1)

add_ = Function(add , "add", 2)
sub_ = Function(sub,  "sub" , 2)
mul_ = Function(mul , "mul", 2)
div_ = Function(div , "div", 2)

sin_ = Function(np.sin , "sin" , 1)
cos_ = Function(np.cos, "cos" , 1)

and_ = Function(np.logical_and, "and" , 2)
or_ = Function(np.logical_or , "or", 2 )
not_ = Function(np.logical_not ,"not",1)
xor_ = Function(np.logical_xor ,"xor" , 2)

CommonFunction = {
    "addOne" : addOne_ ,
    "subOne" : subOne_ ,
    "log" : log_ ,
    "exp" : exp_ ,
    "sqrt" : sqrt_ ,
    "pow2" : pow2_ ,
    "add" : add_,
    "sub" : sub_,
    "mul" : mul_,
    "div" : div_,
    "sin" : sin_,
    "cos" : cos_,

    "and" : and_,
    "or": or_,
    "not":not_,
    "xor": xor_,

}




def get_fitness_MAD(program , X , y):


    y_hat = program.execute(X)

    #print(program.program , y_hat)
    res = np.mean( np.abs( y_hat - y ) )
    return res 




def cal_range( X ):
    return 0

class Program : #binary version
    def __init__(self , terminal_set  , function_set , depth = 3 ,program = None ):
        self.from_method = -1 #init 
        self.terminal_set = terminal_set
        self.function_set = function_set
        self.function_set_arity = {}
        for f in self.function_set:
            if f.arity in self.function_set_arity:
                self.function_set_arity[f.arity].append(f)
            else:
                self.function_set_arity[f.arity] = [f]
        self.depth = depth
        self.fitness = None
        self.values = []
        self.cluster_id = -1
        if program is None :
            self.random_init(self.depth)
            
        else:
            self.program = copy.deepcopy(program)
        # should evaluate when new a program !!!

    def valid(self, program):
        if len(program) == 1 and program[0] < 0 :
            return True
        Counter = [self.function_set[program[0]-1].arity]
        #print("====", program)
        for p in program[1:]:
            #print(program)
            #print(p , Counter)    
            
            if p < 0:
                Counter[-1] -= 1
            else:
                Counter[-1] -= 1
                Counter.append( self.function_set[p-1].arity )
            while Counter !=[] and Counter[-1] == 0:
                Counter.pop()
        return (Counter == [])

    def get_cut_point(self ):
        candidate = [  i for i in range(len(self.program)) if self.program[i] >= 0 
                        and i!=0 
                       ]
        
        if candidate == []:
            return 0
        else:
            return random.choice(candidate)
    
    def execute(self,X):
        """
        value  :  tree length list , each element is each input result at i 
        x = [0 , 1]
        [+ x x]
        [[0,2] ,[0,1] ,[0,1]]
        """
        if( not self.valid(self.program)):
            print("WROOOOOONNNNNG")
            return None
        self.terminals = []
        self.ranges = []
        self.values = []
        output = []
        self.get_parent_forall()
        for p in reversed(self.program):
           
            #print(p , self.program  )
           
                #print("--",i , self.values )
                #print(len(output[i]))
            tmp = None
            if p < 0 : # from terminal set (one dim now!!)
                
                    tmp = X[: , -(p+1)]
                    output.append(tmp)
            else:
                    #print(p,i,output , self.program)
                    if self.function_set[p-1].arity == 1:
                        tmp = self.function_set[p-1](self.values[-1])
                        output[-1] = tmp

                        
                    if self.function_set[p-1].arity == 2:
                        
                        
                        tmp = self.function_set[p-1](output[-1],output[-2])
                        output[-2] = tmp
                        output.pop()
         
                            
            self.values.append(tmp)
          
           
        
           
        self.values = list(reversed(self.values))
        #print(self.values)
        self.y_hat = self.values[0]
        self.ranges = [ -1] * len(self.values)
        for i in range(1,len(self.program)):
            self.ranges[i] =  cal_range(self.values[i])
        self.intron_removal()
        
        
        return self.y_hat
    
    def intron_removal(self , at_i = 0):
        #return
        # for subtree of subtree
        # ****[---[+++]---]****
        #      ^   ^      ^
        #     at_i new_root     end_i
        if len(self.values) == 1:
            return 
        
        root_value = self.values[at_i]
        #print("==", root_value)
        new_root = at_i
        end_i = self.get_subtree(at_i)
        for i in range(len(self.values[at_i : self.get_subtree(at_i) ])):
            if np.array_equal(self.values[at_i + i],root_value):
                #print(at_i + i)
                new_root = at_i + i
        # update termianls
        
        #print(new_root , len(self.program))

        end = self.get_subtree(new_root)+1
        

        if end_i+1 < len(self.values):
            self.values = self.values[:at_i] + self.values[new_root: end] + self.values[end_i+1:]
            self.program = self.program[:at_i] + self.program[new_root: end] + self.program[end_i+1:]
        else:
            self.values = self.values[:at_i] + self.values[new_root: end] 
            self.program = self.program[:at_i] + self.program[new_root: end] 
        
    def get_parent_forall(self):
        self.parentIdx = [-1 for i in range(len(self.program))]
        
        if len(self.program) == 1:
            return 
        
        IdxStack = []
        remain = []
        for i in range(len(self.program)):
            if self.program[i] < 0:
                self.parentIdx[i] = IdxStack[-1] 
                remain[-1] -= 1
                while remain[-1] == 0:
                    remain.pop()
                    IdxStack.pop()
                    if len(remain) == 0:
                        break
                    remain[-1] -= 1
            else:
                if IdxStack != []:
                    self.parentIdx[i] = IdxStack[-1] 
                IdxStack.append(i)
                remain.append( self.function_set[ self.program[i]-1 ].arity )

    def get_path_range(self,i):
       
        if abs( max(self.values[i]) - min(self.values[i]) ) != 0:
            return abs( max(self.values[0]) - min(self.values[0])) / abs( max(self.values[i]) - min(self.values[i]) )
        else:
            return 0


    def random_init(self , depth = 3):
        t = len(self.terminal_set)
        f = len(self.function_set)
        #ramp half and half 
        p = random.uniform(0, 1)
        self.program = []
        if p < 2:# full
            d = 0
            Counter = []

            choice = random.randint(1,f)
            self.program.append(choice)
            
            Counter.append(self.function_set[choice-1].arity)
            while (Counter != []):
                if len(Counter) < depth :
                    Counter[-1] -= 1
                    choice = random.randint(1,f)
                    self.program.append(choice)
                    Counter.append(self.function_set[choice-1].arity)
                else:
                    Counter[-1] -= 1
                    choice = -1 * random.randint(1,t)
                    self.program.append(choice)
                    while Counter[-1] == 0:
                        Counter.pop()
                        if len(Counter) == 0:
                            break

        else:
            d = 0
            not_terminal = True
            while (d < depth and not_terminal) :
                if d < 2:
                    p = 0
                elif d != depth - 1:
                    p = random.uniform(0, 1)
                else:
                    p = 1 # must terminate !! 
                if p < 0.9: #function
                    self.program.append(random.randint(1,f))
                else:
                    not_terminal = False
                    self.program.append(-1 * random.randint(1,t))
                d+=1

    def printX(self):
        #print("\n",self.program ,self.fitness)
        if len(self.program) == 1:
            print("X" + str(-(self.program[0]+1)) , end=" ")
            return 
        Counter = []
        for p in self.program:
            if p < 0:
                Counter[-1] -= 1
                print("X" + str(-(p+1)) , end=" ")
            else:
                if len(Counter) != 0 :
                    Counter[-1] -= 1
                Counter.append(self.function_set[p-1].arity)
                print(self.function_set[p-1].name , "(", end=" ")
            while Counter != [] and Counter[-1] == 0:
                Counter.pop()
                print(")",end=" ")
        print("\nend\n")

    def get_subtree(self , i):
   
        if self.program[i] < 0:
            return i
        counter = self.function_set[self.program[i]-1].arity 
        cur = i
       
        while counter != 0:
            #print(counter,self.program[cur])
            cur += 1
            counter -= 1
            if self.program[cur] >= 0:
                counter += self.function_set[self.program[cur]-1].arity 
        return cur
    def _copy(self,p):
        self.program = copy.deepcopy(p.program)
        self.fitness = p.fitness
        self.values = copy.deepcopy(p.values)
        self.parentIdx = copy.deepcopy(p.parentIdx)
        
        
        
            
        
function_set = [ CommonFunction["add"] 
                ,  CommonFunction["sub"]
                ,  CommonFunction["mul"] 
                , CommonFunction["div"]
                ]

def BinaryToDecimal( binary ):
    n = "".join(binary)
    #print(n)
    return int(n,2)

#from itertools import product
#X = list(product([1, 0], repeat=5) )
# [[true , true , true ] ...  ] 

X = np.linspace(-1,1)

X = np.array( [ np.array([np.array(x)]) for x in X] )
print(X)
# 2**n 個input , output   [1,0,0,1 ...] #len = 2**n
# 2**(2**n)種class

import json 

output = []
id = 0
for d in range(1,5):
    print( d,(len(function_set))**(2**d)   )
    all_binary = (len(function_set))**(2**d)  
  
    num = 10*d**3
    print(num)
    #num = 10*d**3
    for _ in range( num ):
        
        output.append({})
        test = Program( [ i for i in range(len(X[0])) ] , function_set , depth=d)
        output[-1]["program"] = test.program
        output[-1]["output"] = list(test.execute(X))

        #output[-1]["output"] = [ int(i) for i in output[-1]["output"] ]
        #output[-1]["output"] = BinaryToDecimal( output[-1]["output"])  

        output[-1]["id"] = id
        id += 1




print("total num of data:" , len(output))

print( X ,output[2])
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)       
with open("GP_rand_tree.json", "w") as f:
    json.dump(output, f ,cls=NpEncoder)
"""
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
X = [ output[i]["output"] for i in range(len(output)) ] 

silhouette_avg = []
labels = []
for i in range(2,11):
    kmeans_fit = KMeans(n_clusters = i).fit(X)
    silhouette_avg.append(silhouette_score(X, kmeans_fit.labels_))
    labels.append(kmeans_fit.labels_)
print(silhouette_avg)
print(labels)

for i in range(len(output)):
    output[i]["label"] = int(labels[ np.argmax(silhouette_avg) ][i])
    
print([ output[i]["label"] for i in range(len(output)) ] )


"""