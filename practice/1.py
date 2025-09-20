#classes and objects
class model:
    def __init__(self, name, parameter): 
        self.name= name
        self.parameter= parameter
    def summary(self):
        return f"Model name: {self.name}, parameters: {self.parameter}"

model1= model("AetherV1", 1000)
print(model1.summary())