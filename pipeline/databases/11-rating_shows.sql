-- List all shows with the sum of their ratings ordered by rating descending
SELECT tv_shows.title, 
       IFNULL(SUM(ratings.rating), 0) AS rating
FROM tv_shows
LEFT JOIN ratings ON tv_shows.id = ratings.show_id
GROUP BY tv_shows.title
ORDER BY rating DESC;
