#classes and objects
class model: 
    def __init__(self, name, parameter): #here we define the attributes and __init__ is a constructor and self refers to the current instance of the class
        self.name= name
        self.parameter= parameter
    def summary(self):
        return f"Model name: {self.name}, parameters: {self.parameter}"

model1= model("AetherV1", 1000)
print(model1.summary())

#inheritance
class llm_model(model):
    def __init__(self, name, vocab_size):
        self.vocab_size= vocab_size
        super().__init__(name, vocab_size) #super() is used to call the parent class constructor
    def model_run(self): #method overriding- redefining a method in the child class that already exists in the parent class
        return f"Running LLM model: {self.name} with vocab size: {self.vocab_size}"

llm1= llm_model("AETHER-LLM", 100213)
print(llm1.model_run())

#polymorphism
def model_info(model):
    print("polymorphism")
    print(model.summary())
    print(model.model_run())

model_info(llm1)

#encapsulation
class agent:
    def __init__(self,secret_key):
        self.secret_key= secret_key #private attribute
    def get_key(self): #public method to access private attribute
        return self.secret_key
agent1= agent("abcd")
print(agent1.get_key())

#abstraction
from abc import ABC, abstractmethod
class abstract_model(ABC):
    @abstractmethod #abstract method decorator- indicates that the method is abstract and must be implemented in any subclass
    def train(self): #abstract method
        pass
class concrete_model(abstract_model):
    def train(self): #implementing the abstract method
        return "Training the concrete model"

concrete1= concrete_model()
print(concrete1.train())
#decorator
def model_decorator(func):
    def wrapper(*args, **kwargs):
        print("Before Model execution")
        result= func(*args, **kwargs)
        print("After model execution")
        return result
    return wrapper

@model_decorator
def run_model(name):
    return f"Running model: {name}"
print(run_model("AetherV2"))

# iterators and generators
# An iterator is an object that lets you loop through data one element at a time.
# It remembers its position in the sequence → so it doesn’t need all data in memory at once.
# Built using two special methods:
# __iter__() → returns the iterator itself.
# __next__() → returns the next value (or raises StopIteration when done).  
class ModelIterator:
    def __init__(self,model):
        self.model= model 
        self.index= 0
    def __iter__(self): #returns the iterator object itself
            return self
    def __next__(self):
            if self.index < len(self.model): #this means if the index is less than the length of the model list then return the current element
                result=  self.model[self.index]
                self.index +=1
                return result
            else:
                raise StopIteration #when the index is equal to the length of the model list then raise StopIteration
    def reset(self): #reset the iterator to the beginning 
            self.index= 0        
    def reverse(self): #reverse the iterator
            self.model= self.model[::-1]
            self.reset()
models= ['AetherV1', 'AetherV2', 'Aether-LLM']
model_iterator= ModelIterator(models)
print("Iterating through models:")
for model in models:
    print("Model:")
    print(model)
    print("Reversed models:")
    model_iterator.reverse()
    for model in model_iterator:
        print(model)
#generator(simple)
def model_generator(models):
     for model in models:
        yield model
#token stream generator- 
def token_stream_generator(sentence):
    for token in sentence.split():
          yield token
    return "ENd"
def batch_generator(data, batch_size):
    for i in (0, len(data), batch_size):
        yield data[i:i+ batch_size]
    


    