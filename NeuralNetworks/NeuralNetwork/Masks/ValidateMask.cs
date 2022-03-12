using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Masks
{
    internal class ValidateMask : IMask
    {
        public ValidateMask()
        {
            //NTBD
        }

        public Matrix<double> Propagate(Matrix<double> input, double keepProbability)
        {
            return input;
        }
        public Matrix<double> BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            return upstreamWeightedErrors;
        }
    }
}
