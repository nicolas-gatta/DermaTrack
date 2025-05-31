from django.urls import path

from . import views

urlpatterns = [
    path("show_models", views.show_models, name = "show_models"),
    path("model_form", views.model_form, name = "model_form"),
    path("test_model", views.test_model_view, name = "test_model"),
    path("apply_test_sr/<str:image_name>", views.apply_test_sr, name = "apply_test_sr"),
    path("apply_sr/", views.apply_sr, name = "apply_sr"),
    path("load_test_model/<str:model_name>", views.load_test_model,  name = "load_test_model"),
    path("training_model", views.training_model, name = "training_model"),
    path("dataset_form", views.dataset_form, name = "dataset_form"),
    path("evaluation_form", views.evaluation_form, name = "evaluation_form"),
    path("evaluate_model", views.evaluate_model, name = "evaluate_model"),
    path("get_datasets/<str:category>", views.get_datasets,  name = "get_datasets"),
    path("get_test_images", views.get_all_test_images,  name = "get_test_images"),
    path("get_models", views.get_models,  name = "get_models"),
    path("create_dataset", views.create_dataset,  name = "create_dataset"),
    path("get_test_image/<str:name>", views.get_test_image,  name = "get_test_image"),
    path("degrade_and_save_image/<str:name>/<int:scale>", views.degrade_and_save_image,  name = "degrade_and_save_image"),

]