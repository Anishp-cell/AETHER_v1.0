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