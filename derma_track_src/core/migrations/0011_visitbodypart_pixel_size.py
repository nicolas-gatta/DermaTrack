# Generated by Django 5.1.4 on 2025-05-16 19:04

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("core", "0010_alter_visitbodypart_distance_from_subject"),
    ]

    operations = [
        migrations.AddField(
            model_name="visitbodypart",
            name="pixel_size",
            field=models.FloatField(default=0.0014),
        ),
    ]
