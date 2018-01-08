import data as data

class BinaryPerceptron(object):
    def __init__(self, examples, iterations):
        self.w = {}
        for i in xrange(iterations):
            for d, sign in examples:
                prediction = 0.0
                for key in d:
                    prediction += d[key] * self.w.get(key, 0)
                if (sign and prediction <= 0) or (not sign and prediction > 0):
                    for key in d:
                        if sign > 0:
                            self.w[key] = self.w.get(key, 0) + d[key]
                        else:    
                            self.w[key] = self.w.get(key, 0) - d[key]     

    def predict(self, x):
        res = 0
        for key in x:
            res += self.w.get(key, 0) * x[key]
        return res > 0

class MulticlassPerceptron(object):
    def __init__(self, examples, iterations):
        self.d = {label : {} for dict, label in examples }
        # pick a first label
        first = examples[0][1]
        for i in xrange(iterations):
            for dict, l in examples:
                # compute max predicted label
                best_l, best_val = first, None
                for l_key in self.d:
                    prediction = 0
                    d_l = self.d[l_key]                     
                    for key in dict:
                        prediction += d_l.get(key, 0) * dict[key]
                    if best_val == None or prediction > best_val:
                        best_val, best_l = prediction, l_key
                # update weight vector
                if best_l != l:
                    for key in dict:
                        x = dict[key]
                        #Increase the score for the correct class
                        self.d[l][key] = self.d[l].get(key, 0) + x
                        #Decrease the score for the predicted class
                        self.d[best_l][key] = self.d[best_l].get(key, 0) - x
                        
    def predict(self, x):
        best_l = None
        best_val = 0
        for l_key in self.d: # for each label
            prediction = 0
            for data in x: # for each value in dictionary
                prediction += self.d[l_key].get(data, 0) * x[data]
            if prediction > best_val:
                best_val = prediction
                best_l = l_key
        return best_l
    
#Test for binary
#train = [({"x1": 1}, True), ({"x2": 1}, True), ({"x1": -1}, False), ({"x2": -1}, False)]
#test = [{"x1": 1}, {"x1": 1, "x2": 1}, {"x1": -1, "x2": 1.5}, {"x1": -0.5, "x2": -2}]
#p = BinaryPerceptron(train, 1)
#print [p.predict(x) for x in test]

#train = [({"x1": 1}, 1), ({"x1": 1, "x2": 1}, 2), ({"x2": 1}, 3), ({"x1": -1, "x2": 1}, 4), ({"x1": -1}, 5), ({"x1": -1, "x2": -1}, 6),
#         ({"x2": -1}, 7), ({"x1": 1, "x2": -1}, 8)]
#p = MulticlassPerceptron(train, 10)
#print [p.predict(x) for x, y in train]

############################################################
# Application implementations
############################################################

class IrisClassifier(object):

    def __init__(self, data):
        train = list()
        for tup, category in data:
            # length and width of the sepals and petals of the specimen
           d = {"l_sepal": tup[0], "w_sepal": tup[1], "l_petal": tup[2], "w_petal": tup[3]} 
           train.append((d, category))
        self.p = MulticlassPerceptron(train, 2)

    def classify(self, instance):
        d = {"l_sepal": instance[0], "w_sepal": instance[1], "l_petal": instance[2], "w_petal": instance[3]} 
        return self.p.predict(d)
    
#import time
#t = time.time()
#c = IrisClassifier(data.iris)
#print time.time()-t
#count = 0.0
#correct = 0.0
#for tup, category in data.iris:
#    count += 1.0
#    pred = c.classify(tup)
#    if pred == category:
#        correct += 1  
#print correct / count * 100
#print time.time() - t

class DigitClassifier(object):
    def __init__(self, data):
        train = [({ i + 1 : tup[i] for i in xrange(64) }, category) for tup, category in data]
        self.p = MulticlassPerceptron(train, 9)

    def classify(self, instance):
        d = { i + 1 : instance[i] for i in xrange(64) }
        return self.p.predict(d)

#import time
#t = time.time()  
#c = DigitClassifier(data.digits)
#print (time.time() - t)
#count, correct = 0.0, 0.0
#for tup, category in data.digits:
#    count += 1.0
#    pred = c.classify(tup)
#    if pred == category:
#       correct += 1  
#print (correct / count) * 100
#print time.time() - t

class BiasClassifier(object):
    def __init__(self, data):
        train = list()
        for v, bool in data:
            d = {"val": v}
            if v >= 1:
                d["in"] = 1
            else:
                d["out"] = 1
            train.append((d, bool))
        self.p = BinaryPerceptron(train, 2)

    def classify(self, instance):
        d = {"val": instance}
        if instance >= 1:
            d["in"] = 1
        else:
            d["out"] = 1        
        return self.p.predict(d)

#t = time.time()
#c = BiasClassifier(data.bias)
#print time.time() - t
#print [c.classify(x) for x in (-1, 0, 0.5, 1.5, 2)]
#count = 0.0
#correct = 0.0
#for tup, category in data.bias:
#    count += 1.0
#    pred = c.classify(tup)
#    if pred == category:
#        correct += 1  
#print correct / count * 100.0
#print (time.time() - t)

class MysteryClassifier1(object):
    def __init__(self, data):
        train = list()
        for (x,y), bool in data:
            d = {"x": x, "y": y}             
            r = (x * x) / 4.0 + (y * y) / 4.0
            if r >= 1:
                d["in"] = 1
            else:
                d["out"] = 1
            train.append((d, bool))
        self.p = BinaryPerceptron(train, 2)

    def classify(self, instance):
        x, y = instance
        d = {"x": x, "y": y}             
        r = (x * x) / 4.0 + (y * y) / 4.0
        if r >= 1:
            d["in"] = 1
        else:
            d["out"] = 1
        return self.p.predict(d)
#print "mystery1"
import time
t = time.time()
c = MysteryClassifier1(data.mystery1)
print (time.time() - t)
#print [c.classify(x) for x in ((0, 0), (0, 1), (-1, 0), (1, 2), (-3, -4))]
count = 0.0
correct = 0.0
for tup, category in data.mystery1:
    count += 1.0
    pred = c.classify(tup)
    if pred == category:
        correct += 1  
print correct / count * 100
print (time.time() - t)


class MysteryClassifier2(object):

    def __init__(self, data):
        train = list()
        for (x,y,z), bool in data:
            d = {"x": x, "y": y, "z": z}             
            if self.mystery_verifier(x, y, z):
                d["in"] = 1
            else:
                d["out"] = 1
            train.append((d, bool))
        self.p = BinaryPerceptron(train, 4)

    def classify(self, instance):
        x, y, z = instance
        d = {"x": x, "y": y, "z": z}             
        if self.mystery_verifier(x, y, z):
            d["in"] = 1
        else:
            d["out"] = 1
        return self.p.predict(d)
    
    def mystery_verifier(self, x, y, z):
        if 0 <= x and 0 >= y and 0 >= z:
            return True
        elif 0 >= x and 0 >= y and 0 <= z:
            return True
        elif 0 <= x and 0 <= y and 0 <= z:
            return True
        elif 0 >= x and 0 <= y and 0 >= z:
            return True
        else:
            return False        

#t = time.time()
#c = MysteryClassifier2(data.mystery2)
#print (time.time() - t)
#print [c.classify(x) for x in ((1, 1, 1), (-1, -1, -1), (1, 2, -3), (-1, -2, 3))]
#count = 0.0
#correct = 0.0
#for tup, category in data.mystery2:
#    count += 1.0
#    pred = c.classify(tup)
#    if pred == category:
#        correct += 1  
#print correct / count * 100.0
#print (time.time() - t)

