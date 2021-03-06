/****** Script for SelectTopNRows command from SSMS  ******/
SELECT dd.[CarID]
      ,dd.[CodeID]
	  ,bb.[Description]
	  ,dd.cnt 
	  
	  from

(SELECT [CarID]
      ,[CodeID]
	  , Count(CodeID) cnt
  FROM [dbo].[RawData]
  where CarID=10
  group by [CarID],[CodeID]
 ) dd, Codes bb
 where dd.CodeID=bb.ID
 order by cnt desc