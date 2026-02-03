import numpy as np #Numpy is used to declare a array in python
def numpy_usage():
    print("Declaring an array by using numpy:")
    input=np.array([1,2,3,4,5])
    #Broadcasting : Means multiplying entire array with a aspecific value it can be done by uisng * instead of looping through the entire array
    m=input*10
    print(m)
    #Matrix declration:To declare a matrix by using numpy we sue random.rand where it takes single or two arguments indicating size of array
    mat=np.random.rand(5)
    print(mat)
    #dot product
    #Dot product is nothing but multiplying two matrices
    d=np.dot(mat,input)
    print(d)
    #Analysing data
    print(f"Mean value of input{np.mean(input):.2f}")
    print(f"maximumvalue of input{np.max(input)}")
    #Changing dimesnssions of array
    k=input.reshape(5,1)
    print(k)
if __name__=="__main__":
    numpy_usage()