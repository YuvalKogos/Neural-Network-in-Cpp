//////////CREATOR : Yuval Kogos///////////
//////////A Neural Network class in c++ for better spped and precision - using only Vector library/////////

#include "stdafx.h"
#include <vector>
#include <random>
#include <stdlib.h>
#include <iostream>
#include <stdexcept>
#include <math.h>

using namespace std;

double RandomDouble(double min, double max)
{
	random_device rd;
	mt19937 mt(rd());
	uniform_real_distribution<double> dist(-1.0, 1.0);
	return dist(mt);
	//return M + (rand() / (RAND_MAX / (N - M)));
}


double Sigmoid(double val)
{
	return 1 / (1 + exp(-val));
}



class Matrix
{
	//Matrix class for the Linear algebra math and matrix multiplcations
public:
	int rows, cols;
	vector<vector<double>> values;
	Matrix()
	{
		for (int i = 0; i < rows; i++)
		{
			vector<double> tmp;
			for (int j = 0; j < cols; j++)
				tmp.push_back(0.0);
		
			values.push_back(tmp);
		}
	}
	Matrix(vector<vector<double>> values)
	{
		this->values = values;
		this->rows = values.size();
		this->cols = values[0].size();
	}
	Matrix(int r,int c, bool random = true,int default_value = 0.0)
	{
		rows = r;
		cols = c;
		for (int i = 0; i < rows; i++)
		{
			vector<double> tmp;
			for (int j = 0; j < cols; j++)
			{
				if (random)
					tmp.push_back(RandomDouble(-1, 1));
				else
					tmp.push_back(default_value);
			}
			values.push_back(tmp);
		}
	}

	void SetValue(int r, int c, double value)
	{
		values[r][c] = value;
	}

	void Print()
	{
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
				cout << values[i][j] << " , ";
			cout << endl;
		}
		cout << endl;
	}


	static Matrix Substruct(Matrix m1, Matrix m2)
	{
		if (m1.rows != m2.rows || m1.cols != m2.cols)
			throw invalid_argument("Wrong dimensions in add function.");

		Matrix result(m1.rows, m1.cols);

		for (int i = 0; i < m1.rows; i++)
		{
			for (int j = 0; j < m1.cols; j++)
				result.values[i][j] = m1.values[i][j] - m2.values[i][j];
		}

		return result;
	}

	Matrix ActivationFunction()
	{
		Matrix result(rows, cols);
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
			{
				result.values[i][j] = Sigmoid(values[i][j]);
			}
		}

		return result;
	}

	Matrix Transpose()
	{
		Matrix result(this->cols, this->rows);
		for (int i = 0; i < rows; i++)
		{
			for (int j = 0; j < cols; j++)
				result.values[j][i] = values[i][j];
			
		}

		return result;
	}

	static Matrix Add(Matrix m1,Matrix m2)
	{
		if (m1.rows != m2.rows || m1.cols != m2.cols)
			throw invalid_argument("Wrong dimensions in add function.");

		Matrix result(m1.rows, m1.cols);

		for (int i = 0; i < m1.rows; i++)
		{
			for (int j = 0; j < m1.cols; j++)
				result.values[i][j] = m1.values[i][j] + m2.values[i][j];
		}

		return result;
	}

	static Matrix MultiplyByConst(Matrix m1, int k)
	{
		Matrix result(m1.rows, m1.cols);

		for (int i = 0; i < m1.rows; i++)
		{
			for (int j = 0; j < m1.cols; j++)
				result.values[i][j] = m1.values[i][j] * k;
		}

		return result;
	}

	static Matrix MultiplyByConst(Matrix m1, double k)
	{
		Matrix result(m1.rows, m1.cols);

		for (int i = 0; i < m1.rows; i++)
		{
			for (int j = 0; j < m1.cols; j++)
				result.values[i][j] = m1.values[i][j] * k;
		}

		return result;
	}

	static Matrix Multiply(Matrix m1, Matrix m2)
	{
		Matrix product(m1.rows, m2.cols);
		for (int i = 0; i < m1.rows; i++) 
		{
			for (int j = 0; j < m2.cols; j++) 
			{
				for (int k = 0; k < m1.cols; k++) 
				{
					product.values[i][j] = m1.values[i][k] * m2.values[k][j];
				}
			}
		}
		return product;
	}




};

class NeuralNetwork 
{
private:
	int input_nodes, hidden_nodes, output_nodes;
	double learning_rate;
	Matrix IHWeigths, HOWeights,HBias,OBias;
public:
	NeuralNetwork(int inputn, int hiddenn, int outputn, double lr)
	{
		//***Currently supports only 1 hidden-layer
		
		input_nodes = inputn;
		hidden_nodes = hiddenn;
		output_nodes = outputn;
		learning_rate = lr;
		IHWeigths = Matrix(hidden_nodes, input_nodes);
		HOWeights = Matrix(output_nodes, hidden_nodes);
		HBias = Matrix(hidden_nodes, 1);
		OBias = Matrix(output_nodes, 1);


	}
	void PrintWeights()
	{
		//Print Input-Hidden weights
		IHWeigths.Print();	
		//Print Hidden-Output weights
		HOWeights.Print();
		//Print hidden bias
		HBias.Print();
		//Print output bias
		OBias.Print();
	}

	Matrix Predict(Matrix inputs)
	{
		 
		if (inputs.cols > 1 || inputs.rows != IHWeigths.rows)
			throw invalid_argument("Input vector dimension doesn't match the network weight matrix dimensions.");

		//Making a prediction for an input matrix using Feed-Forward algorithm
		
		Matrix IHOutput = Matrix::Multiply(IHWeigths, inputs);
		IHOutput = Matrix::Add(IHOutput, HBias);
		IHOutput.ActivationFunction();
		//Hidden output product
		Matrix HOOutput = Matrix::Multiply(HOWeights, IHOutput);
		HOOutput = Matrix::Add(HOOutput, OBias);
		HOOutput.ActivationFunction();

		return HOOutput;

	}

	void Train(Matrix inputs, Matrix target)
	{
		//Training the net using Back-Propogation technique and Gradient descent algorithm
		
		if (inputs.cols > 1 || inputs.rows != IHWeigths.rows)
			throw invalid_argument("Input vector dimension doesn't match the network weight matrix dimensions.");

		//Input hidden product
		Matrix hidden_pred = Matrix::Multiply(IHWeigths, inputs);
		hidden_pred = Matrix::Add(hidden_pred, HBias);
		hidden_pred.ActivationFunction();
		//Hidden output product
		Matrix output_pred = Matrix::Multiply(HOWeights, hidden_pred);
		output_pred = Matrix::Add(output_pred, OBias);
		output_pred.ActivationFunction();

		//Calculate output errors
		Matrix output_errors = Matrix::Substruct(target, output_pred);

		//Calculate the gradient (for deltas)
		//Gradient = errors * prediction * (1- prediction)

		Matrix gradients = Matrix::Substruct(output_pred, Matrix::Multiply(output_pred, output_pred));
		gradients = Matrix::Multiply(gradients, output_errors);
		gradients = Matrix::MultiplyByConst(gradients, this->learning_rate);

		//Multiply the gradients by the hidden weights transposed
		//to get the delta weights metrices (same dimensions)
		Matrix HOWeights_deltas = Matrix::Multiply(gradients, hidden_pred.Transpose());
		this->HOWeights = Matrix::Add(this->HOWeights, HOWeights_deltas);
		this->OBias = Matrix::Add(OBias, gradients);

		//Calculate the hidden-errors
		Matrix HOWeights_T = HOWeights.Transpose(); /////CHECK!!!////
		Matrix hidden_errors = Matrix::Multiply(HOWeights_T, output_errors);

		//Calculate the hidden gradients
		Matrix gradients_hidden = Matrix::Substruct(hidden_pred, Matrix::Multiply(hidden_pred, hidden_pred));
		gradients_hidden = Matrix::Multiply(gradients, hidden_errors);
		gradients_hidden = Matrix::MultiplyByConst(gradients_hidden, this->learning_rate);

		//Calculate the hidden deltas 
		Matrix IHweights_deltas = Matrix::Multiply(gradients_hidden, inputs.Transpose());

		//Adjust the weights based on the deltas
		this->IHWeigths = Matrix::Add(IHWeigths, IHweights_deltas);
		this->HBias = Matrix::Add(HBias, gradients_hidden);

	}
};



