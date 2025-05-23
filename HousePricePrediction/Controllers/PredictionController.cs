using Microsoft.AspNetCore.Mvc;
using HousePricePrediction.Models;

namespace HousePricePrediction.Controllers
{
    public class PredictionController : Controller
    {
        private readonly MLService _mlService;

        public PredictionController(MLService mlService)
        {
            _mlService = mlService;
        }

        public IActionResult Index()
        {
            return View();
        }

        [HttpPost]
        public IActionResult Predict(HouseData input)
        {
            if (!ModelState.IsValid)
            {
                return View("Index", input);
            }

            var predictedPrice = _mlService.PredictPrice(input);
            ViewBag.PredictedPrice = predictedPrice.ToString("C");
            return View("Index", input);
        }
    }
}