## پارامترها:
- **batch_size** (اندازه دسته)
- **learning_rate** (نرخ یادگیری)

## متریک:
- **دقت** (accuracy)

نمودار مختصات موازی نمایشی از رابطه بین پارامترهای `batch_size` و `learning_rate` با متریک `accuracy` است. مقیاس رنگی در سمت راست نمودار، مقادیر دقت را از 0.81 تا 1.4 نشان می‌دهد. خطوط مختلف در نمودار نمایانگر دو اجرای مختلف هستند که نشان می‌دهند چگونه تغییرات در مقادیر پارامترها روی دقت مدل تأثیر می‌گذارند.

## نتیجه‌گیری:
این نمودار برای درک بهتر این که چگونه تغییرات در پارامترهای `batch_size` و `learning_rate` روی دقت مدل تأثیر می‌گذارند، مفید است. به‌ویژه با توجه به نتایج مقایسه‌ای میان دو اجرا، می‌توان مشاهده کرد که در کدام تنظیمات پارامتر، دقت بهبود یافته است. این نوع بصری‌سازی می‌تواند به تسهیل تصمیم‌گیری در انتخاب بهترین ترکیب پارامترها برای بهبود عملکرد مدل کمک کند.



# Model Lifecycle Documentation

## Model Registration
- Models are registered in the MLflow Model Registry after achieving satisfactory performance on the test dataset.
- Registration happens through the `mlflow.register_model()` API.

## Model Transition Criteria
- **Staging to Production**: A model is transitioned to "Production" if it meets the following criteria:
  - Accuracy greater than 80% on the test dataset.
  - No major issues identified in performance tests.
  
## Versioning Process
- Each new training run that improves model performance or features results in a new version.
- Model versions are compared based on metrics like accuracy, and the latest stable version is promoted.

## Logs and Observations
- A detailed log is maintained with accuracy, training details, and metrics for each model version.


