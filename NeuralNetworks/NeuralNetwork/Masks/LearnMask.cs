using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Masks
{
    internal class LearnMask : IMask
    {
        private Matrix<double> Mu;

        public Random R { get; set; }

        public LearnMask()
        {
            R = new Random();
        }

        public Matrix<double> Propagate(Matrix<double> input, double keepProbability)
        {
            Mu = Matrix<double>.Build.Dense(input.RowCount, input.ColumnCount, 0.0);
            Mu.MapInplace((a) =>
            {
                double random = R.NextDouble();
                if (random < keepProbability) return 1 / keepProbability;
                else return a;
            });
            return input.PointwiseMultiply(Mu);
        }

        public Matrix<double> BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            return upstreamWeightedErrors.PointwiseMultiply(Mu);
        }
    }
}
