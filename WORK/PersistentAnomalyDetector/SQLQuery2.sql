/****** Script for SelectTopNRows command from SSMS  ******/
SELECT [CarID], Count([CarID]) cnt
  FROM [dbo].[RawData]
  GROUP BY [CarID]
  order by cnt DESC

  