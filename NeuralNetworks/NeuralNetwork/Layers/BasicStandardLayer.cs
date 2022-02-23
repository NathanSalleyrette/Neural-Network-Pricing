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

        private Matrix<double> BiasAfterBatch;

        public int BatchSize { get => _batchsize; 
            set { _batchsize = value;
                BiasAfterBatch = InitialBias.Multiply(Matrix<double>.Build.Dense(1, value, 1.0));
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
            InitialBias = initialBias;
            GradientAdjustment = grad;
            BiasAfterBatch = InitialBias.Multiply(Matrix<double>.Build.Dense(1, batchSize, 1.0));
        }

        public void Propagate(Matrix<double> input)
        {
            Input = input;
            NetInput = InitialWeights.Transpose() * input + BiasAfterBatch;

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
            // Seul les weight sont touchés par la penalty de la L2 regularization
            InitialWeights = InitialWeights + Gradient.VWeight((double)(1.0 / M) *  Input * B.Transpose());
            BiasAfterBatch +=  Gradient.VBias((double)(1.0 / M) * B);

            InitialBias = BiasAfterBatch.FoldColumns<double>(
                (s, x) => s + x, Vector<double>.Build.Dense(BiasAfterBatch.RowCount, 0.0))
                .Multiply((double) 1/ BiasAfterBatch.ColumnCount)
                .ToColumnMatrix();


        }

        // Surchage, bonne idée ? Duplication de code pas ouf
        // Fonctionne uniquement pour la reg L2, demander comment améliorer cela
        public void UpdateParameters(double penalty)
        {
            // on multiplie par 1.0 / BatchSize ou par Input.ColumnCount
            // Seul les weight sont touchés par la penalty de la L2 regularization
            InitialWeights = InitialWeights * (1 - penalty * Gradient.LearningRate) + Gradient.VWeight((double)(1.0 / M) * Input * B.Transpose());
            BiasAfterBatch += Gradient.VBias((double)(1.0 / M) * B);

            InitialBias = BiasAfterBatch.FoldColumns<double>(
                (s, x) => s + x, Vector<double>.Build.Dense(BiasAfterBatch.RowCount, 0.0))
                .Multiply((double)1 / BiasAfterBatch.ColumnCount)
                .ToColumnMatrix();
        }

        public bool Equals(ILayer other)
        {
            throw new NotImplementedException();
        }
    }
}