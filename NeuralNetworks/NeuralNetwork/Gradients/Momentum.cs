using System;
using System.Collections.Generic;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using System.Text;
using MathNet.Numerics.LinearAlgebra;

namespace NeuralNetwork.Gradients
{
    public class Momentum : IGradient
    {

        private double learningRate;

        private double momentum;

        private Matrix<double> vBias;

        private Matrix<double> vWeight;
        public Momentum(MomentumParameters gradient, Matrix<double> weight, Matrix<double> bias, int batchSize)
        {
            learningRate = gradient.LearningRate;
            momentum = gradient.Momentum;
            vWeight = Matrix<double>.Build.Dense(weight.RowCount, weight.ColumnCount, 0.0);
            vBias = Matrix<double>.Build.Dense(bias.RowCount, bias.ColumnCount, 0.0);

        }
        public Func<Matrix<double>, Matrix<double>> VBias => (mat) => mat.Multiply(-learningRate);

        public Func<Matrix<double>, Matrix<double>> VWeight => (mat) => mat.Multiply(-learningRate);

        public GradientAdjustmentType Type => GradientAdjustmentType.FixedLearningRate;
    }
}


