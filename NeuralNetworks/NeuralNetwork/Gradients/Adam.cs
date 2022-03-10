using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Common.Serialization;

namespace NeuralNetwork.Gradients
{
    public class Adam : IGradient
    {
        public double StepSize { get; set; }
        public double NumericalStabilizer { get; set; }
        public double ExponentialDecayRateOne { get; set; }
        public double ExponentialDecayRateTwo { get; set; }

        public Matrix<double> RWeight { get; set; }
        public Matrix<double> RPrimeWeight { get; set; }
        public Matrix<double> RBias { get; set; }
        public Matrix<double> RPrimeBias { get; set; }

        public Matrix<double> SWeight { get; set; }
        public Matrix<double> SPrimeWeight { get; set; }
        public Matrix<double> SBias { get; set; }
        public Matrix<double> SPrimeBias { get; set; }
        public Matrix<double> vWeight { get; set; }
        public Matrix<double> vBias { get; set; }
        public int i { get; set; }

        public Adam(AdamParameters gradient, Matrix<double> weight, Matrix<double> bias, int batchSize)
        {
            StepSize = gradient.StepSize;
            NumericalStabilizer = gradient.DenominatorFactor;
            ExponentialDecayRateOne = gradient.FirstMomentDecay;
            ExponentialDecayRateTwo = gradient.SecondMomentDecay;
            

            vWeight = Matrix<double>.Build.Dense(weight.RowCount, weight.ColumnCount, 0.0);
            vBias = Matrix<double>.Build.Dense(bias.RowCount, bias.ColumnCount, 0.0);


            RWeight = Matrix<double>.Build.Dense(weight.RowCount, weight.ColumnCount, 0.0);
            RPrimeWeight = Matrix<double>.Build.Dense(weight.RowCount, weight.ColumnCount, 0.0);
            RBias = Matrix<double>.Build.Dense(bias.RowCount, bias.ColumnCount, 0.0);
            RPrimeBias = Matrix<double>.Build.Dense(bias.RowCount, bias.ColumnCount, 0.0);


            SWeight = Matrix<double>.Build.Dense(weight.RowCount, weight.ColumnCount, 0.0);
            SPrimeWeight = Matrix<double>.Build.Dense(weight.RowCount, weight.ColumnCount, 0.0);
            SBias = Matrix<double>.Build.Dense(bias.RowCount, bias.ColumnCount, 0.0);
            SPrimeBias = Matrix<double>.Build.Dense(bias.RowCount, bias.ColumnCount, 0.0);
            vWeight = Matrix<double>.Build.Dense(weight.RowCount, weight.ColumnCount, 0.0);
            vBias = Matrix<double>.Build.Dense(bias.RowCount, bias.ColumnCount, 0.0);


            i = 0;
        }
        public Func<Matrix<double>, Matrix<double>> VWeight => (mat) =>
        {
            i++;
            SWeight = SWeight.Multiply(ExponentialDecayRateOne) + mat.Multiply(1 - ExponentialDecayRateOne);
            RWeight = RWeight.Multiply(ExponentialDecayRateTwo) + mat.PointwiseMultiply(mat).Multiply(1 - ExponentialDecayRateTwo);

            SPrimeWeight = SWeight.Multiply(1.0 / (1 - Math.Pow(ExponentialDecayRateOne, i)));
            RPrimeWeight = RWeight.Multiply(1.0 / (1 - Math.Pow(ExponentialDecayRateTwo, i)));

            RPrimeWeight = RPrimeWeight.PointwiseSqrt().Add(NumericalStabilizer);

            vWeight = SPrimeWeight.PointwiseDivide(RPrimeWeight).Multiply(-StepSize);

            return vWeight;
        };
        public Func<Matrix<double>, Matrix<double>> VBias => (mat) =>
        {
            SBias = SBias.Multiply(ExponentialDecayRateOne) + mat.Multiply(1 - ExponentialDecayRateOne);
            RBias = RBias.Multiply(ExponentialDecayRateTwo) + mat.PointwiseMultiply(mat).Multiply(1 - ExponentialDecayRateTwo);

            SPrimeBias = SBias.Multiply(1.0 / (1 - Math.Pow(ExponentialDecayRateOne, i)));
            RPrimeBias = RBias.Multiply(1.0 / (1 - Math.Pow(ExponentialDecayRateTwo, i)));

            RPrimeBias = RPrimeBias.PointwiseSqrt().Add(NumericalStabilizer);

            vBias = SPrimeBias.PointwiseDivide(RPrimeBias).Multiply(-StepSize);

            return vBias;
        };

        public double LearningRate => throw new NotImplementedException();

        public GradientAdjustmentType Type => GradientAdjustmentType.Adam;
    }
}
