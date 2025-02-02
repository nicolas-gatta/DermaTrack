from django.urls import path

from . import views

urlpatterns = [
    path("show_models", views.show_models, name = "show_models"),
    path("model_form", views.model_form, name = "model_form"),
    path("test_model", views.test_model_view, name = "test_model"),
    path("training_model", views.training_model, name = "training_model"),
    path("dataset_form", views.dataset_form, name = "dataset_form"),
    path("get_datasets/<str:category>", views.get_datasets,  name = "get_datasets"),
    path("get_test_images", views.get_test_images,  name = "get_test_images"),
    path("get_models", views.get_models,  name = "get_models"),
    path("create_dataset", views.create_dataset,  name = "create_dataset"),
    path("get_test_image/<str:name>", views.get_test_image,  name = "get_test_image"),
    path("degrade_and_save_image/<str:name>/<int:scale>", views.degrade_and_save_image,  name = "degrade_and_save_image"),
    path("test", views.test, name = "test"),
    path("test_2", views.test_2, name = "test_2"),
    path("test_3", views.test_3, name = "test_3"),
    path("test_4", views.test_4, name = "test_4")
]