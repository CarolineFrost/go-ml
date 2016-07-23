package main
import "fmt"
import "math"

// linearRegression implements batch gradient descent with a cost function that
// is consistent with ordinary least squares
func linearRegression(featureMatrix [][]float64, targetArray []float64, convergence float64) (arr []float64){
    if len(featureMatrix) == 0 || len(featureMatrix) != len(targetArray) {
        return nil
    }
    numFeatures := len(featureMatrix[0])
    numTrainingEx := len(featureMatrix)
    theta := make([] float64, numFeatures)
    stepSize := .0005
    var error float64
    var priorError float64
    for p := 0; p < 1e6; p++ {
        error = costFunction(featureMatrix, targetArray, theta)
        if math.Abs(error - priorError) <= convergence {return theta}
        priorError = error
        for i := 0; i < numFeatures; i++ {
            var sum float64
            for j := 0; j < numTrainingEx; j++ {
                sum += (targetArray[j] - hypothesisFunction(featureMatrix[j], theta)) * featureMatrix[j][i]   

            }
            theta[i] += stepSize * sum
        }
    }

    panic(fmt.Sprintf("linearRegression did not converge"))
}

func hypothesisFunction(featureMatrix []float64, theta []float64) (h float64) {
    numFeatures := len(featureMatrix)
    for j := 0; j < numFeatures; j++ {
        h += featureMatrix[j] * theta[j]
    }
    return
}

func costFunction(featureMatrix [][]float64, targetArray []float64, theta []float64) (error float64) {
    numFeatures := len(featureMatrix[0])
    numTrainingEx := len(featureMatrix)
    for i := 0; i < numTrainingEx; i++ {
        var innersum float64
        for j := 0; j < numFeatures; j++ {
            innersum += featureMatrix[i][j] * theta[j]
        }
        error += math.Pow((targetArray[i] - innersum) , 2)
    }
    return
}
