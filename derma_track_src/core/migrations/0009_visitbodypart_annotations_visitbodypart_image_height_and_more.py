# Generated by Django 5.1.4 on 2025-05-16 14:28

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        (
            "core",
            "0008_rename_distance_from_subjet_visitbodypart_distance_from_subject",
        ),
    ]

    operations = [
        migrations.AddField(
            model_name="visitbodypart",
            name="annotations",
            field=models.TextField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name="visitbodypart",
            name="image_height",
            field=models.IntegerField(default=0),
        ),
        migrations.AddField(
            model_name="visitbodypart",
            name="image_width",
            field=models.IntegerField(default=0),
        ),
        migrations.AlterField(
            model_name="visit",
            name="is_patient_present",
            field=models.BooleanField(blank=True, default=False),
        ),
    ]
