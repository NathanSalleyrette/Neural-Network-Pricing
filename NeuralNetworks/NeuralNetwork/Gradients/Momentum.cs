using System;
using System.Collections.Generic;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetwork.Gradients
{
    public class Momentum : IGradient
    {

        public double LearningRate { get; }

        private double momentum;

        private Matrix<double> vBias;

        private Matrix<double> vBiasAfterBatch;

        private Matrix<double> vWeight;
        public Momentum(MomentumParameters gradient, Matrix<double> weight, Matrix<double> bias, int batchSize)
        {
            LearningRate = gradient.LearningRate;
            momentum = gradient.Momentum;
            vWeight = Matrix<double>.Build.Dense(weight.RowCount, weight.ColumnCount, 0.0);
            
            // Matrice Colonne
            vBias = Matrix<double>.Build.Dense(bias.RowCount, bias.ColumnCount, 0.0);

        }
        public Func<Matrix<double>, Matrix<double>> VBias => (mat) =>
        {
            vBiasAfterBatch = vBias.Multiply(Matrix<double>.Build.Dense(1, mat.ColumnCount, 1.0));
            vBiasAfterBatch = vBiasAfterBatch.Multiply(momentum) - mat.Multiply(LearningRate);
            
            // Mean of Columns to have a vector of Bias
            vBias = vBiasAfterBatch.FoldColumns<double>(
               (s, x) => s + x,
               Vector<double>.Build.Dense(vBiasAfterBatch.RowCount, 0.0))
                .Multiply((double)1 / vBiasAfterBatch.ColumnCount).ToColumnMatrix();
            
            return vBiasAfterBatch;
        };
        public Func<Matrix<double>, Matrix<double>> VWeight => (mat) => 
        {
           vWeight = vWeight.Multiply(momentum) - mat.Multiply(LearningRate);
           return vWeight;
        };

        public GradientAdjustmentType Type => GradientAdjustmentType.Momentum;
    }
}


