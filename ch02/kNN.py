# coding=UTF-8
from numpy import *
import operator
from os import listdir


def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    ##假设dataSet m行n列
    dataSetSize = dataSet.shape[0] ##获取dataSet的行数dataSetSize，也即m
    ##inX本来是1行n列，将其扩展成为m行n列，并与dataSet相减，得到差值对应的矩阵
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat ** 2
     ##每一行的值相加，压缩成只有1列，而不是n列
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    ## 将distances从小到大排序，而后按照这个顺序，返回各个值对应的索引
    sortedDistIndicies = distances.argsort() 
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        ##key=operator.itemgetter(1)实际上生成了一个函数，
        # key=operator.itemgetter(1)表示按照classCount的第二个域的值进行排序
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1))
    ##返回结果值最小的那个样本的class类型
    return sortedClassCount[0][0]

# 将文件中的内容转换成特征值矩阵、类型向量
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])

def autoNorm(dataSet):
    # dataSet中每一行的最小值、最大值
    # 0 表示从行这个维度来寻找最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals## m行1列的一个特殊矩阵
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.50
    datingDataMat,datingLabels = file2matrix('/home/fmr/workbench/code_py/machineLearning/ch02/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d, the read answer is: %d" % (classifierResult, datingLabels[i])
        if(classifierResult != datingLabels[i]):
            errCount += 1.0
    print "the total error rate is: %f" % (errCount/float(numTestVecs))
    print errCount        

# def handwritingClassTest():
#     hwLabels = []
#     trainingFileList = listdir('/home/fmr/workbench/code_py/machineLearning/ch02/trainingDigits')
#     m = len(trainingFileList)
#     trainingMat = zeros((m,1024))
#     for i in range(m):
#         fileNameStr = trainingFileList[i]
#         fileStr = fileNameStr.split('.')[0]
#         classNumStr = int(fileStr.split('_')[0])
#         hwLabels.append(classNumStr)
#         # 该文件夹下，每个文件转换成一个img
#         trainingMat[i,:] = img2vector('/home/fmr/workbench/code_py/machineLearning/ch02/trainingDigits/%s' % fileNameStr)
#     testFileList = listdir('/home/fmr/workbench/code_py/machineLearning/ch02/testDigits')
#     errorCount = 0.0
#     mTest = len(testFileList)
#     for i in range(mTest):
#         fileNameStr = testFileList[i]
#         fileStr = fileNameStr.split('.')[0]
#         classNumStr = int(fileStr.split('_')[0])
#         vectorUnderTest = img2vector('/home/fmr/workbench/code_py/machineLearning/ch02/testDigits/%s' % fileNameStr)
#         classifierResult = classify0(vectorUnderTest,trainingMat,hwLabels,3)
#         print "the classifier came back with: %d, the real answer is: %d" %(classifierResult,classNumStr)
#         if(classifierResult != classNumStr):
#             errCount += 1.0
#         print "\nthe total number of errors is: %d" % errCount
#         print "\nthe total error rate is: %f" % (errCount/float(mTest))

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('/home/fmr/workbench/code_py/machineLearning/ch02/trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('/home/fmr/workbench/code_py/machineLearning/ch02/trainingDigits/%s' % fileNameStr)
    testFileList = listdir('/home/fmr/workbench/code_py/machineLearning/ch02/testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('/home/fmr/workbench/code_py/machineLearning/ch02/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))


def main():
    #datingClassTest()
    handwritingClassTest()

if(__name__=="__main__"):
        main()