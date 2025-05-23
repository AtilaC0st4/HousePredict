using Microsoft.ML.Data;

namespace HousePricePrediction.Models
{
    public class HouseData
    {
        [LoadColumn(0)] public float Size;  // Não usado no modelo?
        [LoadColumn(1)] public float SqFt;
        [LoadColumn(2)] public float Bedrooms;
        [LoadColumn(3)] public float Bathrooms;
        [LoadColumn(4)] public float Neighborhood;  // Tipo numérico!
        [LoadColumn(5)] public float YearBuilt;
        [LoadColumn(6)] public float Price;
    }
    public class HousePrediction : HouseData
    {
        [ColumnName("Score")]
        public float PredictedPrice;
    }
}