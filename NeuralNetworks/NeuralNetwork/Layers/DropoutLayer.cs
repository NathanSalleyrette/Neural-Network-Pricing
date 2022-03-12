using MathNet.Numerics.LinearAlgebra;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.Layers;
using NeuralNetwork.Gradients;
using System;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using MathNet.Numerics.Distributions;
using MathNet.Numerics.Random;
using NeuralNetwork.Common;
using NeuralNetwork.Masks;



namespace NeuralNetwork.Layers
{
    class DropoutLayer : ILayer, IComponentWithMode
    {

        // Weights of the layer : Id
        // Bias of the layer    : 0
        public int LayerSize {get; set;}

        public double KeepProbability { get; set; }

        public int InputSize { get; set; }

        public int BatchSize { get; set; }

        public Matrix<double> Activation { get; set; }

        public Matrix<double> WeightedError { get; set; }

        // Ceci est notre masque
        //public Matrix<double> Mu { get; set; }

        public Random R { get; set; }

        // Le mode définit comment va se comporter notre masque
        // si on est en mode Training, alors un masque est appliqué
        // En mode Validation, le masque n'enlève aucune neurone
        private Mode mode;
        public Mode Mode { 
            get { return mode; } 
            set {
                mode = value;
                if (value == Mode.Training) Mask = new LearnMask();
                else Mask = new ValidateMask();
                } 
        }

        public IMask Mask { get; set; }

        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            WeightedError = Mask.BackPropagate(upstreamWeightedErrors);
        }   

        public bool Equals(ILayer other)
        {
            throw new NotImplementedException();
        }

        public void Propagate(Matrix<double> input)
        { 
            Activation = Mask.Propagate(input, KeepProbability);
        }

        public void UpdateParameters()
        {
            // NTBD
        }

        public DropoutLayer(int layerSize, double keepProbability, int batchSize)
        {
            LayerSize = layerSize;
            InputSize = layerSize;

            KeepProbability = keepProbability;
            BatchSize = batchSize;
            // On crée une mat activation que l'on va changer dans propagate
            Activation = Matrix<double>.Build.Dense(1, 1);
            WeightedError = Matrix<double>.Build.Dense(1, 1);

            R = new Random();

        }
    }
}
