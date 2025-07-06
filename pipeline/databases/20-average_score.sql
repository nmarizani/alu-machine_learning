-- Drop procedure if it already exists to prevent errors
DROP PROCEDURE IF EXISTS ComputeAverageScoreForUser;

DELIMITER $$

-- Create the procedure
CREATE PROCEDURE ComputeAverageScoreForUser (
    IN in_user_id INT
)
BEGIN
    DECLARE avg_score FLOAT;

    -- Calculate the average score from corrections for the user
    SELECT AVG(score) INTO avg_score
    FROM corrections
    WHERE user_id = in_user_id;

    -- Update the average_score in the users table
    UPDATE users
    SET average_score = IFNULL(avg_score, 0)
    WHERE id = in_user_id;
END$$

DELIMITER ;
