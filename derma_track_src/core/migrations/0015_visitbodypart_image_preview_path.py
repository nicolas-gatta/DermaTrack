# Generated by Django 5.1.4 on 2025-05-17 12:00

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("core", "0014_alter_visitbodypart_image_path"),
    ]

    operations = [
        migrations.AddField(
            model_name="visitbodypart",
            name="image_preview_path",
            field=models.ImageField(default="", unique=False, upload_to=""),
        ),
    ]
