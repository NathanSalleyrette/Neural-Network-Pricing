using MathNet.Numerics.LinearAlgebra;


namespace NeuralNetwork.Masks
{
    internal interface IMask
    {
        Matrix<double> Propagate(Matrix<double> input, double keepProbability);

        Matrix<double> BackPropagate(Matrix<double> upstreamWeightedErrors);
    }
}
