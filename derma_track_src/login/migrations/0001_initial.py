# Generated by Django 5.1.4 on 2025-01-18 16:27

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name="Doctor",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("national_number", models.CharField(max_length=11, unique=True)),
                ("INAMI", models.CharField(max_length=255, unique=True)),
                ("name", models.CharField(max_length=255)),
                ("surname", models.CharField(max_length=255)),
                ("date_of_birth", models.DateField()),
                ("street", models.CharField(max_length=255)),
                ("number", models.IntegerField()),
                ("city", models.CharField(max_length=255)),
                ("zip_code", models.IntegerField()),
                ("phone_number", models.CharField(max_length=255)),
                ("other_phone_number", models.CharField(blank=True, max_length=255)),
                ("is_retired", models.BooleanField()),
                (
                    "user",
                    models.OneToOneField(
                        blank=True,
                        on_delete=django.db.models.deletion.CASCADE,
                        to=settings.AUTH_USER_MODEL,
                    ),
                ),
            ],
        ),
    ]
