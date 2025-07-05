-- List all genres with the sum of their ratings ordered descending
SELECT tv_genres.name, 
       IFNULL(SUM(ratings.rating), 0) AS rating
FROM tv_genres
JOIN tv_show_genres ON tv_genres.id = tv_show_genres.genre_id
JOIN tv_shows ON tv_show_genres.show_id = tv_shows.id
LEFT JOIN ratings ON tv_shows.id = ratings.show_id
GROUP BY tv_genres.name
ORDER BY rating DESC;
