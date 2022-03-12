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
        public Matrix<double> RBiasColumn { get; set; }

        public Matrix<double> RPrimeBias { get; set; }

        public Matrix<double> SWeight { get; set; }
        public Matrix<double> SPrimeWeight { get; set; }
        public Matrix<double> SBias { get; set; }
        public Matrix<double> SBiasColumn { get; set; }

        public Matrix<double> SPrimeBias { get; set; }
        public Matrix<double> VMatWeight { get; set; }
        public Matrix<double> VMatBias { get; set; }
        public int I { get; set; }

        public Adam(AdamParameters gradient, Matrix<double> weight, Matrix<double> bias, int batchSize)
        {
            StepSize = gradient.StepSize;
            NumericalStabilizer = gradient.DenominatorFactor;
            ExponentialDecayRateOne = gradient.FirstMomentDecay;
            ExponentialDecayRateTwo = gradient.SecondMomentDecay;
            

            //vWeight = Matrix<double>.Build.Dense(weight.RowCount, weight.ColumnCount, 0.0);
            //vBias = Matrix<double>.Build.Dense(bias.RowCount, bias.ColumnCount, 0.0);


            RWeight = Matrix<double>.Build.Dense(weight.RowCount, weight.ColumnCount, 0.0);
            //RPrimeWeight = Matrix<double>.Build.Dense(weight.RowCount, weight.ColumnCount, 0.0);
            RBiasColumn = Matrix<double>.Build.Dense(bias.RowCount, bias.ColumnCount, 0.0);
            //RPrimeBias = Matrix<double>.Build.Dense(bias.RowCount, bias.ColumnCount, 0.0);


            SWeight = Matrix<double>.Build.Dense(weight.RowCount, weight.ColumnCount, 0.0);
            //SPrimeWeight = Matrix<double>.Build.Dense(weight.RowCount, weight.ColumnCount, 0.0);
            
            SBiasColumn = Matrix<double>.Build.Dense(bias.RowCount, bias.ColumnCount, 0.0);
            //SPrimeBias = Matrix<double>.Build.Dense(bias.RowCount, bias.ColumnCount, 0.0);
            //vWeight = Matrix<double>.Build.Dense(weight.RowCount, weight.ColumnCount, 0.0);
            //vBias = Matrix<double>.Build.Dense(bias.RowCount, bias.ColumnCount, 0.0);


            I = 0;
        }
        public Func<Matrix<double>, Matrix<double>> VWeight => (mat) =>
        {
            I++;
            SWeight = SWeight.Multiply(ExponentialDecayRateOne) + mat.Multiply(1 - ExponentialDecayRateOne);
            RWeight = RWeight.Multiply(ExponentialDecayRateTwo) + mat.PointwiseMultiply(mat).Multiply(1 - ExponentialDecayRateTwo);

            SPrimeWeight = SWeight.Multiply(1.0 / (1 - Math.Pow(ExponentialDecayRateOne, I)));
            RPrimeWeight = RWeight.Multiply(1.0 / (1 - Math.Pow(ExponentialDecayRateTwo, I)));

            RPrimeWeight = RPrimeWeight.PointwiseSqrt().Add(NumericalStabilizer);

            VMatWeight = SPrimeWeight.PointwiseDivide(RPrimeWeight).Multiply(-StepSize);

            return VMatWeight;
        };
        public Func<Matrix<double>, Matrix<double>> VBias => (mat) =>
        {
            SBias = SBiasColumn.Multiply(Matrix<double>.Build.Dense(1, mat.ColumnCount, 1.0));
            SBias = SBias.Multiply(ExponentialDecayRateOne) + mat.Multiply(1 - ExponentialDecayRateOne);

            SBiasColumn = SBias.FoldColumns<double>(
               (s, x) => s + x,
               Vector<double>.Build.Dense(SBias.RowCount, 0.0))
                .Multiply((double)1 / SBias.ColumnCount).ToColumnMatrix();

            RBias = RBiasColumn.Multiply(Matrix<double>.Build.Dense(1, mat.ColumnCount, 1.0));
            RBias = RBias.Multiply(ExponentialDecayRateTwo) + mat.PointwiseMultiply(mat).Multiply(1 - ExponentialDecayRateTwo);

            RBiasColumn = RBias.FoldColumns<double>(
                (s, x) => s + x,
                 Vector<double>.Build.Dense(RBias.RowCount, 0.0))
                    .Multiply((double)1 / RBias.ColumnCount).ToColumnMatrix();

            SPrimeBias = SBias.Multiply(1.0 / (1 - Math.Pow(ExponentialDecayRateOne, I)));
            RPrimeBias = RBias.Multiply(1.0 / (1 - Math.Pow(ExponentialDecayRateTwo, I)));

            RPrimeBias = RPrimeBias.PointwiseSqrt().Add(NumericalStabilizer);

            VMatBias = SPrimeBias.PointwiseDivide(RPrimeBias).Multiply(-StepSize);

            return VMatBias;
        };

        public double LearningRate => throw new NotImplementedException();

        public GradientAdjustmentType Type => GradientAdjustmentType.Adam;
    }
}
