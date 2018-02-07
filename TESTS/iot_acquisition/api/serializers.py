from rest_framework import serializers

class CarDataSerializer(serializers.Serializer):
    CarID = serializers.IntegerField()
    Code = serializers.CharField(max_length=2)
    Value = serializers.IntegerField()