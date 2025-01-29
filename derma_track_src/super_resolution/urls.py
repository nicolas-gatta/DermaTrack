from django.urls import path

from . import views

urlpatterns = [
    path("training", views.training_srcnn, name="index"),
    path("show_models", views.show_models, name = "show_models"),
    path("model_form", views.model_form, name = "model_form"),
    path("training_model", views.training_model, name = "training_model"),
    path("dataset_form", views.dataset_form, name = "dataset_form"),
    path("get_datasets/<str:category>", views.get_datasets,  name = "get_datasets"),
    path("create_dataset", views.create_dataset,  name = "create_dataset"),
    path("test", views.test, name = "test"),
    path("test_2", views.test_2, name = "test_2"),
    path("test_3", views.test_3, name = "test_3"),
    path("test_4", views.test_4, name = "test_4")
]