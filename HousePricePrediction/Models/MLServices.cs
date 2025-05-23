using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;

namespace HousePricePrediction.Models
{
    public class MLService
    {
        private MLContext _mlContext;
        private ITransformer _model;

        public MLService()
        {
            _mlContext = new MLContext();

            // Carrega os dados
            var data = _mlContext.Data.LoadFromTextFile<HouseData>(
                path: Path.Combine(Directory.GetCurrentDirectory(), "Data", "houses.csv"),
                hasHeader: true,
                separatorChar: ',');

            // Divide os dados 70/30
            var split = _mlContext.Data.TrainTestSplit(data, testFraction: 0.3);

            // Define o pipeline de treinamento
            var pipeline = _mlContext.Transforms
     .Concatenate("Features",  // Neighborhood já é numérica (1, 2, 3)
         nameof(HouseData.SqFt),
         nameof(HouseData.Bedrooms),
         nameof(HouseData.Bathrooms),
         nameof(HouseData.Neighborhood),  // Usada diretamente como número
         nameof(HouseData.YearBuilt))
     .Append(_mlContext.Transforms.NormalizeMinMax("Features"))  // Normalização
     .Append(_mlContext.Regression.Trainers.FastTree(
         labelColumnName: nameof(HouseData.Price),
         numberOfLeaves: 10,   // Reduzido para evitar overfitting
         numberOfTrees: 50,    // Reduzido para dataset pequeno
         minimumExampleCountPerLeaf: 2));  // Menor devido ao pouco dados

            // Treina o modelo
            _model = pipeline.Fit(split.TrainSet);


            // Avalia o modelo
            var predictions = _model.Transform(split.TestSet);
            var metrics = _mlContext.Regression.Evaluate(predictions, "Price");

            Console.WriteLine($"R² Score: {metrics.RSquared}");
        }

        public float PredictPrice(HouseData input)
        {
            var predictionEngine = _mlContext.Model.CreatePredictionEngine<HouseData, HousePrediction>(_model);
            var prediction = predictionEngine.Predict(input);
            return prediction.PredictedPrice;
        }
    }
}