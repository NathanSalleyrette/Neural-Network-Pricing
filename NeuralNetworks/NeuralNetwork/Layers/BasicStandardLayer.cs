using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Gradients;
using System;
using NeuralNetwork.Common.GradientAdjustmentParameters;

namespace NeuralNetwork.Layers
{
    internal class BasicStandardLayer : ILayer
    {

        public int InputSize { get; }
        public int LayerSize { get; }

        private int _batchsize;

        private int M { get; }

        private Matrix<double> BiasBeforeBatch;

        public int BatchSize { get => _batchsize; 
            set { _batchsize = value;
                InitialBias = BiasBeforeBatch.Multiply(Matrix<double>.Build.Dense(1, value, 1.0));
                Activation = Matrix<double>.Build.Dense(Activation.RowCount, value);
                B = Matrix<double>.Build.Dense(B.RowCount, value);
                
            } }

        public IActivator Activator { get; }


        public Matrix<double> Activation { get; set; }

        public Matrix<double> WeightedError { get; set;  }

        public Matrix<double> InitialWeights { get; set; }

        public Matrix<double> InitialBias { get; set; }


        public Matrix<double> NetInput { get; set; }

        public Matrix<double> Input { get; set; }

        public Matrix<double> B { get; set; }


        public IGradient Gradient { get;  }

        public IGradientAdjustmentParameters GradientAdjustment { get;  }

        public BasicStandardLayer(Matrix<double> initialWeights, IGradient gradientAdjustment, Matrix<double> initialBias, int batchSize, IActivator activator, IGradientAdjustmentParameters grad)
        {
            
            LayerSize = initialWeights.ColumnCount;
            InputSize = initialWeights.RowCount;
            _batchsize = batchSize;
            Activator = activator ?? throw new ArgumentNullException(nameof(activator));
            Activation = Matrix<double>.Build.Dense(LayerSize, BatchSize);
            NetInput = Matrix<double>.Build.Dense(LayerSize, BatchSize);
            Input = Matrix<double>.Build.Dense(LayerSize, BatchSize);
            M = Input.ColumnCount;
            B = Matrix<double>.Build.Dense(LayerSize, BatchSize);
            Gradient = gradientAdjustment;
            InitialWeights = initialWeights;
            BiasBeforeBatch = initialBias;
            GradientAdjustment = grad;
            InitialBias = BiasBeforeBatch.Multiply(Matrix<double>.Build.Dense(1, batchSize, 1.0));
        }

        public void Propagate(Matrix<double> input)
        {
            Input = input;
            NetInput = InitialWeights.Transpose() * input + InitialBias;

            //NetInput = InitialWeights.Transpose() * input + InitialBias;
            NetInput.Map(Activator.Apply, Activation);
        }

        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            NetInput.Map(Activator.ApplyDerivative, B);
            B.PointwiseMultiply(upstreamWeightedErrors, B);
            WeightedError = InitialWeights * B;

        }

        public void UpdateParameters()
        {
            // on multiplie par 1.0 / BatchSize ou par Input.ColumnCount 
            InitialWeights +=  Gradient.VWeight((double)(1.0 / M) *  Input * B.Transpose());
            InitialBias +=  Gradient.VBias((double)(1.0 / M) * B);
        }

        public bool Equals(ILayer other)
        {
            throw new NotImplementedException();
        }
    }
}