import Foundation

// ------- a math class for some simple math funcs --------------
class math {

    // simple random generator between 0.001 and 1
    func rand() -> Double {return Double(arc4random_uniform(999) + 1)/1000}

    // just a random array generator
    func rand_array(_ units:Int) -> [Double] {
        var result:[Double] = []
        for _ in 1...units {
            result.append(rand())
        }
        return result
    }

    // zero array generator
    func zero_array(_ units:Int) -> [Double] {
        var result:[Double] = []
        for _ in 1...units {
            result.append(0.0)
        }
        return result
    }

    // ones array generator
    func ones_array(_ units:Int) -> [Double] {
        var result:[Double] = []
        for _ in 1...units {
            result.append(1.0)
        }
        return result
    }

    // matrix by matrix multiplication
    func matrix_mul(_ a:matrix,_ b:matrix) -> matrix {
        let a_rows = a.matrix.count
        let a_columns = a.matrix[0].count
        let b_rows = b.matrix.count
        let b_columns = b.matrix[0].count
        let result = matrix()
        result.zeros(a_rows, b_columns)
        if b_rows == a_columns {
            for i in 0...a_rows-1 {
                for j in 0...b_columns-1 {
                    for k in 0...a_columns-1 {
                        result.matrix[i][j] = result.matrix[i][j] + a.matrix[i][k]*b.matrix[k][j]
                    }
                }
            }
        }else{
            print("invalid matrix by matrix multiplication")
        }
        return result
    }

    // matrix by matrix elementwise multiplication
    func matrix_mul_elem(_ a:matrix,_ b:matrix) -> matrix {
        let a_rows = a.matrix.count
        let a_columns = a.matrix[0].count
        let b_rows = b.matrix.count
        let b_columns = b.matrix[0].count
        let result = matrix()
        result.zeros(a_rows, b_columns)
        if a_rows == b_rows && a_columns == b_columns {
            for i in 0...a_rows-1 {
                for j in 0...b_columns-1 {
                    result.matrix[i][j] = result.matrix[i][j] + a.matrix[i][j]*b.matrix[i][j]
                }
            }
        }else{
            print("invalid matrix by matrix elementwise multiplication")
        }
        return result
    }

    // matrix by matrix subtraction
    func matrix_sub(_ a:matrix,_ b:matrix) -> matrix {
        let a_rows = a.matrix.count
        let a_columns = a.matrix[0].count
        let b_rows = b.matrix.count
        let b_columns = b.matrix[0].count
        let result = matrix()
        result.zeros(a_rows, b_columns)
        if b_rows == a_rows && b_columns == a_columns {
            for i in 0...a_rows-1 {
                for j in 0...b_columns-1 {
                    result.matrix[i][j] = a.matrix[i][j] - b.matrix[i][j]
                }
            }
        }else{
            print("invalid matrix by matrix subtraction")
        }
        return result
    }


    // transpose of a matrix
    func transpose(_ input:matrix) -> matrix {
        let rows = input.matrix.count
        let columns = input.matrix[0].count
        let result = matrix()
        result.zeros(columns, rows)
        for i in 0...columns-1 {
            for j in 0...rows-1 {
                result.matrix[i][j] = input.matrix[j][i]
            }
        }
        return result
    }

    // sigmoid function
    func sigmoid(_ input: Double) -> Double{
        return 1/(1+exp(-input))
    }
    func sigmoid_matrix(_ input:matrix) -> matrix {
        let result = input
        for i in 0...input.matrix.count-1 {
            for j in 0...input.matrix[0].count-1{
                result.matrix[i][j] = sigmoid(result.matrix[i][j])
            }
        }
        return result
    }
}

// ------------- An implementation of a matrix class ------------------
class matrix: math {
    var matrix:[[Double]] = []

    func random(_ iNum:Int ,_ jNum:Int) {
        matrix = []
        for _ in 1...iNum {
            matrix.append(rand_array(jNum))
        }
    }

    func zeros(_ iNum:Int ,_ jNum:Int) {
        matrix = []
        for _ in 1...iNum {
            matrix.append(zero_array(jNum))
        }
    }

    func ones(_ iNum:Int ,_ jNum:Int) {
        matrix = []
        for _ in 1...iNum {
            matrix.append(ones_array(jNum))
        }
    }

    func get(_ i:Int, _ j:Int) -> Double {
        return matrix[i][j]
    }

    func get() -> [[Double]] {
        return matrix
    }

    func multiply(with: Double) {
        for i in 0...matrix.count-1{
            for j in 0...matrix[0].count-1{
                matrix[i][j] = with*matrix[i][j]
            }
        }
    }
}

// ------------- An implementation of a simple neural network from scratch ------------------
class NeuralNetwork:math {

    var layers:[matrix] = []
    var layers_z:[matrix] = []
    var thetas:[matrix] = []
    var step = 0.1
    var Error_theta:[matrix] = []
    var Error_layer_z:[matrix] = []

    init(numberOfUnits:[Int]) {
        for i in 0...numberOfUnits.count-1 {
            let mat = matrix(); mat.random(numberOfUnits[i], 1)
            layers.append(mat)
            layers_z.append(mat)
            Error_layer_z.append(mat)
        }

        for i in 0...numberOfUnits.count-2 {
            let mat3 = matrix(); mat3.random(numberOfUnits[i+1], numberOfUnits[i]+1)
            thetas.append(mat3)
            Error_theta.append(mat3)
        }
    }

    func output_for(input:matrix) -> [[Double]]{
        let one_input = math().transpose(input)
        calc_layers(input: one_input)
        return layers[layers.count-1].matrix
    }

    func train(input:matrix, output:matrix, iteration_num: Int){
        for i in 1...iteration_num {
            var iteration_error = 0.0
            for j in 0...input.matrix.count-1 {
                let one_input = matrix(); one_input.random(input.matrix[0].count, 1)
                for k in 0...input.matrix[0].count-1 {
                    one_input.matrix[k][0] = input.matrix[j][k]
                }
                let one_output = matrix(); one_output.random(output.matrix[0].count, 1)
                for k in 0...output.matrix[0].count-1 {
                    one_output.matrix[k][0] = output.matrix[j][k]
                }
                back_propagation(input: one_input, output: one_output)
                iteration_error = calc_backprob_error(output: one_output)
            }
            print("error iteration \(i): \(iteration_error/Double(input.matrix.count))")
        }
    }

    func back_propagation(input: matrix, output: matrix) {
        calc_layers(input: input)
        calc_output_error(output: output)
        calc_layers_error()
        calc_thetas_error()
        weight_update_SGD()
    }

    func calc_layers(input:matrix) {
        layers[0] = input
        for i in 0...layers.count-2 {
            let oneFront = matrix()
            oneFront.matrix = layers[i].matrix
            oneFront.matrix.insert([1.0], at: 0)
            layers_z[i+1] = matrix_mul(thetas[i], oneFront)
            layers[i+1] = math().sigmoid_matrix(layers_z[i+1])
        }
    }

    func sigmoid_derivative(layer:matrix) -> matrix {
        let ones = matrix(); ones.ones(layer.matrix.count, 1)
        let result = math().matrix_mul_elem(layer, math().matrix_sub(ones, layer))
        return result
    }

    func calc_output_error(output:matrix) {
        let aux = math().matrix_sub(layers[layers.count-1], output)
        Error_layer_z[Error_layer_z.count-1] = math().matrix_mul_elem(aux, sigmoid_derivative(layer: layers[layers.count-1]))
    }

    func calc_layers_error() {
        for i in (0...layers.count-2).reversed() {
            let aux = math().matrix_mul(math().transpose(thetas[i]), Error_layer_z[i+1])
            aux.matrix.remove(at: 0)
            Error_layer_z[i] = math().matrix_mul_elem(aux, sigmoid_derivative(layer: layers_z[i]))
        }
    }

    func calc_thetas_error() {
        for i in 0...Error_theta.count-1{
            Error_theta[i] = math().matrix_mul(layers[i], math().transpose(Error_layer_z[i+1]))
        }
    }

    func weight_update_SGD() {
        for i in 0...thetas.count-1{
            let oneFront = matrix()
            oneFront.matrix = layers[i].matrix
            oneFront.matrix.insert([1.0], at: 0)
            let aux = math().matrix_mul(oneFront, math().transpose(Error_layer_z[i+1]))
            aux.multiply(with: step)
            thetas[i] = math().matrix_sub(thetas[i], math().transpose(aux))
        }
    }

    func calc_backprob_error(output:matrix) -> Double {
        var error = 0.0
        for i in 0...layers[layers.count-1].matrix.count-1{
            for j in 0...layers[layers.count-1].matrix[i].count-1{
                let aux = (layers[layers.count-1].matrix[i][j] - output.matrix[i][j])
                error = error + aux*aux
            }
//            error = error/Double(layers[layers.count-1].matrix[i].count)
        }
        error = error/Double(layers[layers.count-1].matrix.count)
        return error
    }

}
