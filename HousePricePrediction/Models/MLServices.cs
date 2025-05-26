using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms;

namespace HousePricePrediction.Models
{
    public class MLService
    {
        private MLContext _mlContext;
        private ITransformer _model;
        private DataViewSchema _modelSchema;

        public MLService()
        {
            _mlContext = new MLContext(seed: 1); // Definir seed para reprodutibilidade

            // Carrega os dados
            var data = _mlContext.Data.LoadFromTextFile<HouseData>(
                path: Path.Combine(Directory.GetCurrentDirectory(), "Data", "houses.csv"),
                hasHeader: true,
                separatorChar: ',');

            // Divide os dados 80/20 (mais dados para treino)
            var split = _mlContext.Data.TrainTestSplit(data, testFraction: 0.2);

            // Define o pipeline de treinamento com normalização individual
            var pipeline = _mlContext.Transforms
                .CustomMapping<HouseData, ToNormalize>(mapAction: (input, output) =>
                {
                    output.SqFt = input.SqFt;
                    output.Bedrooms = input.Bedrooms;
                    output.Bathrooms = input.Bathrooms;
                    output.Neighborhood = input.Neighborhood;
                    output.YearBuilt = input.YearBuilt;
                    output.Price = input.Price;
                }, contractName: "NormalizeMapping")
                .Append(_mlContext.Transforms.NormalizeMinMax("SqFt", "SqFt"))
                .Append(_mlContext.Transforms.NormalizeMinMax("Bedrooms", "Bedrooms"))
                .Append(_mlContext.Transforms.NormalizeMinMax("Bathrooms", "Bathrooms"))
                .Append(_mlContext.Transforms.NormalizeMinMax("Neighborhood", "Neighborhood"))
                .Append(_mlContext.Transforms.NormalizeMinMax("YearBuilt", "YearBuilt"))
                .Append(_mlContext.Transforms.Concatenate("Features",
                    "SqFt", "Bedrooms", "Bathrooms", "Neighborhood", "YearBuilt"))
                .Append(_mlContext.Regression.Trainers.FastTree(
                    labelColumnName: nameof(HouseData.Price),
                    numberOfLeaves: 30,
                    numberOfTrees: 200,
                    minimumExampleCountPerLeaf: 5,
                    learningRate: 0.1));

            // Treina o modelo
            _model = pipeline.Fit(split.TrainSet);
            _modelSchema = split.TrainSet.Schema;

            // Avalia o modelo
            var predictions = _model.Transform(split.TestSet);
            var metrics = _mlContext.Regression.Evaluate(predictions, "Price");

            Console.WriteLine($"R² Score: {metrics.RSquared}");
            Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError}");
        }

        public float PredictPrice(HouseData input)
        {
            // Verifica valores mínimos/máximos
            input.SqFt = Math.Max(800, Math.Min(3000, input.SqFt));
            input.Bedrooms = Math.Max(1, Math.Min(5, input.Bedrooms));
            input.Bathrooms = Math.Max(1, Math.Min(4, input.Bathrooms));
            input.Neighborhood = Math.Max(1, Math.Min(3, input.Neighborhood));
            input.YearBuilt = Math.Max(1950, Math.Min(2023, input.YearBuilt));

            var predictionEngine = _mlContext.Model.CreatePredictionEngine<HouseData, HousePrediction>(_model);
            var prediction = predictionEngine.Predict(input);
            return prediction.PredictedPrice;
        }

        private class ToNormalize
        {
            public float SqFt { get; set; }
            public float Bedrooms { get; set; }
            public float Bathrooms { get; set; }
            public float Neighborhood { get; set; }
            public float YearBuilt { get; set; }
            public float Price { get; set; }
        }
    }
}