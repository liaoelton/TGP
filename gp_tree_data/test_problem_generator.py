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




class Population :
    def __init__ (self , 
                terminal_set , 
                function_set ,
                num_population , 
                generation , 
                data_X,
                data_y, 
                depth = 5 ):
        self.num_population = num_population 
        self.terminal_set = terminal_set
        self.function_set = function_set
        
        
        self.dim = len(data_X[0])
        self.depth = depth
        self.generation = generation
        self.programs = [Program(terminal_set , function_set) for _ in range(self.num_population )  ]
        self.X = data_X
        self.y = data_y
  
        self.minX = [ min([x for x in data_X[:,i]])  for i in range(self.dim) ]
        self.maxX = [ max([x for x in data_X[:,i]])  for i in range(self.dim) ]
        
        self.rangeX = [ abs(self.maxX[i] - self.minX[i]) for i in range(len(self.maxX))  ]




        self.cluster_bigger = [[] , []] # funcs , args
        self.cluster_smaller = [[], []] # funcs , args
        
        self.target_range = abs( max(data_y) - min(data_y) )
        #self.input_range = abs( max(data_X) - min(data_X) )
        #build bigger function
        self.bigger_function = []
        self.smaller_function = []
        """
        idx = 0
      
        for f in function_set:
            if f.arity == 1:
                output = [f(x) for x in self.X]
            
                r = abs( max(output) - min(output) ) / abs( max( data_X ) - min( data_X ) )
            
                if r > 1:
                    self.bigger_function.append(idx)
                elif r < 1: 
                    self.smaller_function.append(idx)
                idx+=1
            elif f.arity == 2:
                output = [f(x,x) for x in self.X]
            
                r = abs( max(output) - min(output) ) / abs( max( data_X ) - min( data_X ) )
            
                if r > 1:
                    self.bigger_function.append(idx)
                elif r < 1: 
                    self.smaller_function.append(idx)
                idx+=1
        """
        for p in self.programs:
            p.fitness = get_fitness_MAD(p , data_X ,data_y)
        
        
    def tournament_selection(self):
        a = random.choice(self.programs)
        b = random.choice(self.programs)
       
        if a.fitness < b.fitness:
            return a
        else:
            return b
  
    def one_point_crossover(self , donor , receiver  ):
        # receiver : Program class
        # donor : Program class
        donor_cutpoint = random.randint(0,len(donor.program)-1)
        receiver_cutpoint = random.randint(0,len(receiver.program)-1)
        new_program =  donor.program[:donor_cutpoint] + receiver.program[receiver_cutpoint:]

        return Program(self.terminal_set , self.function_set , program=new_program)
 
    def subtree_mutation(self , donor):
        new_chicken = Program(self.terminal_set  , self.function_set , self.depth)
        return self.one_point_crossover(donor , new_chicken)

    def run_once(self):
        
        self.next_programs = [ self.programs[i] for i in range(self.num_population)]
        self.parent_func = [ self.programs[i] for i in range(self.num_population)]
        self.parent_args = [ self.programs[i] for i in range(self.num_population)]
        
      
        for i in range(self.num_population):
            p = random.uniform(0, 1)
            if p < 0: # dimension mutation 
                donor = self.tournament_selection()
                terminals = [i for i in range(len(donor.program)) if donor.program[i] < 0   ]
                terminal_i = random.choice(terminals)
               
                t = donor.program[terminal_i]
                t = -(t+1)
                #print(terminal_i , donor.program ,donor.program[terminal_i],donor.terminals , len(donor.program) )
                new_program = copy.deepcopy(donor.program)
                r = donor.get_path_range(terminal_i)
                if  r / self.target_range > 1:
                    choice_i = [ i  for i in range(len(self.rangeX)) if self.rangeX[i] < self.rangeX[t]   ] #need smaller terminal X
                    
                    if choice_i != []:
                        new_term = -(random.choice(choice_i)+1)
                        new_program[terminal_i] = new_term 
                    else:
                 
                        new_program[terminal_i] = -1 * random.randint(1,len(self.terminal_set))
                else:
                    choice_i = [ i for i in range(len(self.rangeX)) if self.rangeX[i] > self.rangeX[t]   ] #need bigger terminal X
                  
                    if choice_i != []:
                        new_term = -(random.choice(choice_i)+1)
                        #print(new_term)
                        new_program[terminal_i] = new_term
                    else:
                       
                        new_program[terminal_i] = -1 * random.randint(1,len(self.terminal_set))
                #print(new_program)
                self.next_programs[i] = Program(self.terminal_set , self.function_set , program=new_program)
                self.next_programs[i].fitness = get_fitness_MAD(self.next_programs[i] , self.X , self.y)
                #print( self.programs[i].fitness , self.next_programs[i].fitness)
                self.next_programs[i].from_method = 0
            elif p < 1:

                #ranger crossover 
                    """
                    donor = self.tournament_selection()
                    receiver = self.tournament_selection()
                    self.programs[i] = self.one_point_crossover(donor , receiver)
                    self.programs[i].fitness = get_fitness_MAD(self.programs[i] , self.X , self.y)
                    """
                    donor = self.tournament_selection()
                    arg_receiver_idx = None
                    arg_receiver_func = None

                    start = donor.cut_point
                    end = donor.get_subtree(donor.cut_point)

                    softmax_prob = None
                    if donor.func_range > 1:
                        softmax_prob = random.uniform(1 - (1/donor.func_range),1)
                    else:
                        softmax_prob = random.uniform(0,donor.func_range) #small

                    if softmax_prob > 0.5:
                        if self.cluster_smaller[1] != []:
                            arg_receiver_idx = random.choice(self.cluster_smaller[1])
                        elif self.smaller_function != []:
                            arg_receiver_func = random.choice(self.smaller_function)
                    

                    else :
                        if self.cluster_bigger[1] != []:
                            arg_receiver_idx = random.choice(self.cluster_bigger[1])
                        elif self.bigger_function != []:
                            arg_receiver_func = random.choice(self.bigger_function)

                    t = len(self.terminal_set)
                    choice_1 = -1 * random.randint(1,t)
                    choice_2 = -1 * random.randint(1,t)


                    # update for mutation
                    self.bigger_function = []
                    self.smaller_function = []
                    _min_x = min(min(self.X[:,-(choice_1+1)]) , min(self.X[:,-(choice_2+1)]))
                    _max_x = max(max(self.X[:,-(choice_1+1)]) , max(self.X[:,-(choice_2+1)]))
                    idx = 0
      
                    for f in function_set:
                        if f.arity == 1:
                            output = [f(x) for x in self.X]
                        
                            r = abs( max(output) - min(output) ) / abs( _max_x - _min_x )
                        
                            if r > 1:
                                self.bigger_function.append(idx)
                            elif r < 1: 
                                self.smaller_function.append(idx)
                            idx+=1
                        elif f.arity == 2:
                            output = f(self.X[:,-(choice_1+1)], self.X[:,-(choice_2+1)])  

                   
                            r = abs( max(output) - min(output) ) / abs( _max_x - _min_x )
                        
                            if r > 1:
                                self.bigger_function.append(idx)
                            elif r < 1: 
                                self.smaller_function.append(idx)
                            idx+=1



                    if arg_receiver_idx != None:
                        receiver = self.programs[arg_receiver_idx]
                        r_start = receiver.cut_point
                        r_end = receiver.get_subtree(receiver.cut_point)

                        new_program =  donor.program[:start] + receiver.program[r_start:r_end+1] + donor.program[end+1:]
                

                    elif arg_receiver_func != None: 
                        
                        new_program =  donor.program[:start] + [ arg_receiver_func,  choice_1 ,  choice_2 ] + donor.program[end+1:]
                        
                    else:
                        random_func = random.randint(0, len(self.function_set)-1)
                        new_program = donor.program[:start] + [ random_func, choice_1, choice_2 ] + donor.program[end+1:]
                    
                    self.next_programs[i] = Program(self.terminal_set , self.function_set , program=new_program)
                    self.next_programs[i].fitness = get_fitness_MAD(self.next_programs[i] , self.X , self.y)
                    self.next_programs[i].from_method = 1
                    
            elif p < 1: #reproduced
                    donor = self.tournament_selection()
                    self.programs[i] = copy.deepcopy(donor)
        
        # replace
        dim_mu_success = 0
        for i in range(self.num_population):
            
            if self.programs[i].fitness > self.next_programs[i].fitness:
                #print("before:",self.parent_func[i] , self.parent_args[i])
                #print("after:",self.next_programs[i].program)
                if self.next_programs[i].from_method == 0:
                    dim_mu_success+=1
                self.programs[i]._copy(self.next_programs[i])
                
        #print("dim_mu_success:" , dim_mu_success)
        #print(success)    
        self.programs = sorted(self.programs, key=lambda p: p.fitness)
    def run(self):
        for g in range(self.generation):
            # build cluster for funcs and args
            _id = 0
            self.cluster_bigger = [[] , []] 
            self.cluster_smaller = [[], []] 

            for p in self.programs:
               
                if len(p.program) == 1:
                    self.programs[_id] = Program(self.terminal_set , self.function_set)
                    self.programs[_id].fitness = get_fitness_MAD(self.programs[_id], self.X , self.y)
                    #print("neeeeeeew")
                p = self.programs[_id]
                p.cut_point = random.randint(1,len(p.program)-1)
                #print(p.cut_point)
                #print(p.program)
                #print(p.program[:p.cut_point] , p.program[p.cut_point:])
                #print( "===", _id , p.range , p.program ,  p.cut_point  )
                
                func_range = p.get_path_range(p.cut_point)   #  path range
                
                #arg_program = p.program[ p.cut_point : p.get_subtree(p.cut_point)+1 ]
                #Max_ = max([ self.maxX[-(a+1)] for a in arg_program if a < 0])
                #min_ = min([ self.minX[-(a+1)] for a in arg_program if a < 0])


                args_range = ( abs( max(p.values[p.cut_point]) - min(p.values[p.cut_point]))
                                / abs( 1 ) )  #  output_range  /  1

                

             
                p.intron_removal(p.cut_point)
                p.get_parent_forall()
               
                
                if (func_range / self.target_range) > 1:
                    p.func_range = func_range / self.target_range
                    self.cluster_bigger[0].append(_id)
                    
                else:
                    p.func_range = func_range / self.target_range
                    self.cluster_smaller[0].append(_id)
                
                #print(args_range , self.target_range)
                if (args_range ) > 1:
                    p.arg_range = 1
                    self.cluster_bigger[1].append(_id)
                else:
                    p.arg_range = -1
                    self.cluster_smaller[1].append(_id)
                #print(_id)
                """
                if len(self.programs[_id].program) != len(self.programs[_id].parentIdx):
                    print("gen",g)
                    print(self.programs[_id].program , self.programs[_id].parentIdx )
                    exit()
                """
                _id += 1

            #print("big" , self.cluster_bigger)
            #print("small" , self.cluster_smaller)
            self.run_once()
            
            #self._print_programs()
           
            
        self.programs = sorted(self.programs, key=lambda p: p.fitness)
    




    def _print_programs(self):
 
        i = 0
        for p in self.programs:
            print(i, p.fitness) 
            p.printX()
            print()
            i+=1    
    



class Program : #binary version
    def __init__(self , terminal_set  , function_set , depth = 5 ,program = None ):
        self.from_method = -1 #init 
        self.terminal_set = terminal_set
        self.function_set = function_set
        self.depth = depth
        self.fitness = None
        self.values = []
        if program is None :
            self.random_init(self.depth)
        else:
            self.program = program
        # should evaluate when new a program !!!

    def valid(self, program):
        if len(program) == 1 and program[0] < 0 :
            return True
        Counter = [self.function_set[program[0]-1].arity]
        #print("====", program)
        for p in program[1:]:
           #print(p , Counter)    
            if p < 0:
                Counter[-1] -= 1
            else:
                Counter[-1] -= 1
                Counter.append( self.function_set[p-1].arity )
            while Counter !=[] and Counter[-1] == 0:
                Counter.pop()
        return (Counter == [])

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
        self.range = []
        self.values = []
        output = []

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

        self.intron_removal()
        self.get_parent_forall()
        
        return self.y_hat
    
    def intron_removal(self , at_i = 0):
    
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
        cur = i
        res = 1
        #print(len(self.parentIdx) , cur)
        while self.parentIdx[cur] != -1:
           
            parent = self.parentIdx[cur]

            if abs( max(self.values[cur]) - min(self.values[cur]) )!= 0:
                res*= ( 
                    abs( max(self.values[parent]) - min(self.values[parent]))
                    / abs( max(self.values[cur]) - min(self.values[cur]) ))
            else:
                res *= 0

            cur = parent 
            
        return res



    def random_init(self , depth = 3):
        t = len(self.terminal_set)
        f = len(self.function_set)
        #ramp half and half 
        p = random.uniform(0, 2)
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


    def get_subtree(self , i):
   
        if self.program[i] < 0:
            return i
        counter = self.function_set[self.program[i]-1].arity 
        cur = i

        while counter != 0:
            cur += 1
            counter -= 1
            if self.program[cur] >= 0:
                counter += self.function_set[self.program[i]-1].arity 
        return cur
    def _copy(self,p):
        self.program = copy.deepcopy(p.program)
        self.fitness = p.fitness
        self.values = copy.deepcopy(p.values)
        self.parentIdx = copy.deepcopy(p.parentIdx)
        
        
            
        
function_set = [ CommonFunction["and"] 
                ,  CommonFunction["or"]
                ,  CommonFunction["xor"] 

                ]

def BinaryToDecimal( binary ):
    n = "".join(binary)
    #print(n)
    return int(n,2)

from itertools import product
X = list(product([1, 0], repeat=4) )
# [[true , true , true ] ...  ] 


X = np.array( [np.array(x) for x in X] )
# 2**n 個input , output   [1,0,0,1 ...] #len = 2**n
# 2**(2**n)種class

import json 

output = []
id = 0
d = 3

test = Program( [ i for i in range(len(X[0])) ] , function_set , depth=d)

print(test.program)
output = list(test.execute(X))
test.intron_removal()
print(test.program)
print(output)
       




"""
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