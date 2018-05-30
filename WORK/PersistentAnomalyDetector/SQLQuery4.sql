/****** Script for SelectTopNRows command from SSMS  ******/
SELECT  [ID]
      ,[CarID]
      ,[CodeID]
      ,[StrValue]
      ,[Value]
      ,[ViewVal]
      ,[ViewStr]
      ,[TimeStamp]
      ,[DeviceSN]
  FROM [dbo].[RawData]
  Where TimeStamp >'2018-01-28' and TimeStamp < '2018-02-02'
  and CarID=57