from django.db import models

# Create your models here.


class Image(models.Model):
    image = models.ImageField(null=False, blank=False)
    date = models.TextField(null=True, blank=True)
    location = models.TextField()

    def __str__(self):
        return self.location


class UserUpload(models.Model):
    image = models.ImageField(null=False, blank=False)
    date = models.TextField(null=True, blank=True)
    location = models.TextField(null=True, blank=True)
    area = models.IntegerField(null=True, blank=True)
    cluster = models.IntegerField(null=False, blank=False, default=2)

    def __str__(self):
        return self.location
