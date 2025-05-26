using Microsoft.ML.Data;
using System.ComponentModel.DataAnnotations;

namespace HousePricePrediction.Models
{
    public class HouseData
    {
        [LoadColumn(0)]
        [Range(800, 3000, ErrorMessage = "A área deve estar entre 800 e 3000 pés quadrados")]
        public float SqFt { get; set; }

        [LoadColumn(1)]
        [Range(1, 5, ErrorMessage = "O número de quartos deve estar entre 1 e 5")]
        public float Bedrooms { get; set; }

        [LoadColumn(2)]
        [Range(1, 4, ErrorMessage = "O número de banheiros deve estar entre 1 e 4")]
        public float Bathrooms { get; set; }

        [LoadColumn(3)]
        [Range(1, 3, ErrorMessage = "O bairro deve ser 1 (Subúrbio), 2 (Cidade) ou 3 (Rural)")]
        public float Neighborhood { get; set; }

        [LoadColumn(4)]
        [Range(1950, 2023, ErrorMessage = "O ano de construção deve estar entre 1950 e 2023")]
        public float YearBuilt { get; set; }

        [LoadColumn(5)]
        public float Price { get; set; }
    }

    public class HousePrediction
    {
        [ColumnName("Score")]
        public float PredictedPrice { get; set; }
    }
}