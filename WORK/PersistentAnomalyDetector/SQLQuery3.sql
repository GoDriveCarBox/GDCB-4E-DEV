/****** Script for SelectTopNRows command from SSMS  ******/
SELECT CarID, CodeID, Count(CodeID) cnt
from RawData
where CarID=57
Group By CarID, CodeID
order by cnt desc
