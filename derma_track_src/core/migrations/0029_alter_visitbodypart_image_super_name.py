# Generated by Django 5.1.4 on 2025-05-31 14:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0028_alter_visitbodypart_image_name_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='visitbodypart',
            name='image_super_name',
            field=models.CharField(blank=True, default='', max_length=255, null=True),
        ),
    ]
