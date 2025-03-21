package pgvector

import (
	"context"
	"fmt"

	"github.com/agent-api/core"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgxpool"
	"github.com/pgvector/pgvector-go"
	pgxvec "github.com/pgvector/pgvector-go/pgx"
)

// PgVectorStore implements the core.VectorStorer interface in agent-api/core.
// It assumes the Postgres DB has the pgvector extension installed.
type PgVectorStore struct {
	pool       *pgxpool.Pool
	tableName  string
	dimensions int
}

// PgVectorStoreOpts holds configuration for initializing a PgVectorStore
type PgVectorStoreOpts struct {
	ConnectionString string
	TableName        string
	Dimensions       int
}

// New creates a new PgVectorStore
func New(ctx context.Context, config *PgVectorStoreOpts) (*PgVectorStore, error) {
	// initial connection to establish "vector" extension
	conn, err := pgx.Connect(ctx, config.ConnectionString)
	if err != nil {
		return nil, fmt.Errorf("failed to make initial connection: %w", err)
	}
	defer conn.Close(ctx)

	// ensure pgvector extension is enabled
	_, err = conn.Exec(ctx, "CREATE EXTENSION IF NOT EXISTS vector")
	if err != nil {
		return nil, fmt.Errorf("failed to create/check vector extension: %w", err)
	}

	// Create connection pool conf
	poolConfig, err := pgxpool.ParseConfig(config.ConnectionString)
	if err != nil {
		return nil, fmt.Errorf("invalid connection string: %w", err)
	}

	// Register vector types using pgx pool conf
	poolConfig.AfterConnect = func(ctx context.Context, conn *pgx.Conn) error {
		return pgxvec.RegisterTypes(ctx, conn)
	}

	// Create the connection pool
	pool, err := pgxpool.NewWithConfig(ctx, poolConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create connection pool: %w", err)
	}

	// Test connection pool
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	store := &PgVectorStore{
		pool:       pool,
		tableName:  config.TableName,
		dimensions: config.Dimensions,
	}

	// Initialize table
	if err := store.initTable(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("failed to initialize table: %w", err)
	}

	return store, nil
}

// initTable creates the vector table if it doesn't exist
func (s *PgVectorStore) initTable(ctx context.Context) error {
	// Create table if it doesn't exist
	query := fmt.Sprintf(`
        CREATE TABLE IF NOT EXISTS %s (
            id TEXT PRIMARY KEY,
            vector vector(%d) NOT NULL,
            content TEXT
        )
    `, pgx.Identifier{s.tableName}.Sanitize(), s.dimensions)

	_, err := s.pool.Exec(ctx, query)
	if err != nil {
		return fmt.Errorf("could not create table: %w", err)
	}

	// Create index for faster similarity search
	indexQuery := fmt.Sprintf(`
        CREATE INDEX IF NOT EXISTS %s ON %s USING ivfflat (vector vector_l2_ops)
        WITH (lists = 100)
    `, pgx.Identifier{s.tableName + "_vector_idx"}.Sanitize(), pgx.Identifier{s.tableName}.Sanitize())

	_, err = s.pool.Exec(ctx, indexQuery)
	if err != nil {
		return fmt.Errorf("could not create table index: %w", err)
	}

	return nil
}

// Add implements VectorStorer.Add in agent-api/core
func (s *PgVectorStore) Add(ctx context.Context, vectors []*core.Embedding) error {
	var err error

	// Start a transaction
	tx, err := s.pool.Begin(ctx)
	if err != nil {
		return fmt.Errorf("could not begin pool transaction: %w", err)
	}

	// Use defer with a named error to handle rollback properly
	defer func() {
		if err != nil {
			tx.Rollback(ctx)
		}
	}()

	// Prepare SQL statement once, outside the loop
	insertSQL := fmt.Sprintf(`
        INSERT INTO %s (id, vector, content) 
        VALUES ($1, $2, $3)
        ON CONFLICT (id) DO UPDATE SET
        vector = EXCLUDED.vector,
        content = EXCLUDED.content
    `, pgx.Identifier{s.tableName}.Sanitize())

	// Prepare batch for efficient insertion
	batch := &pgx.Batch{}

	for _, cv := range vectors {
		// Check vector dimensions
		if len(cv.Vector) != s.dimensions {
			return fmt.Errorf("vector dimension mismatch: expected %d, got %d", s.dimensions, len(cv.Vector))
		}

		// Convert to pgvector type
		vec := pgvector.NewVector(cv.Vector)

		// Add to batch
		batch.Queue(insertSQL, cv.ID, vec, cv.Content)
	}

	// Execute batch
	results := tx.SendBatch(ctx, batch)

	// Check for any errors in the batch execution
	for i := range batch.Len() {
		println("executing batch")
		if _, err = results.Exec(); err != nil {
			return fmt.Errorf("error executing batch at index %d: %w", i, err)
		}
	}

	// close the batch results
	results.Close()

	// Commit transaction
	err = tx.Commit(ctx)
	if err != nil {
		return fmt.Errorf("failed to commit transaction: %w", err)
	}

	return nil
}

// Search implements VectorStorer.Search
func (s *PgVectorStore) Search(ctx context.Context, params core.SearchParams) ([]core.SearchResult, error) {
	// Validate query vector dimensions
	if len(params.Query) != s.dimensions {
		return nil, fmt.Errorf("query vector dimension mismatch: expected %d, got %d", s.dimensions, len(params.Query))
	}

	// Set default limit if not specified
	limit := params.Limit
	if limit <= 0 {
		limit = 10
	}

	// Convert query vector to pgvector type
	queryVec := pgvector.NewVector(params.Query)

	// Build query with threshold if specified
	var rows pgx.Rows
	var err error

	if params.Threshold > 0 {
		query := fmt.Sprintf(`
			SELECT id, vector, content, vector <-> $1 AS distance
			FROM %s
			WHERE vector <-> $1 < $2
			ORDER BY distance
			LIMIT $3
		`, pgx.Identifier{s.tableName}.Sanitize())

		rows, err = s.pool.Query(ctx, query, queryVec, params.Threshold, limit)
	} else {
		query := fmt.Sprintf(`
			SELECT id, vector, content, vector <-> $1 AS distance
			FROM %s
			ORDER BY distance
			LIMIT $2
		`, pgx.Identifier{s.tableName}.Sanitize())

		rows, err = s.pool.Query(ctx, query, queryVec, limit)
	}

	if err != nil {
		return nil, err
	}
	defer rows.Close()

	// Process results
	var results []core.SearchResult
	for rows.Next() {
		var (
			id       string
			vec      pgvector.Vector
			content  string
			distance float32
		)

		if err := rows.Scan(&id, &vec, &content, &distance); err != nil {
			return nil, err
		}

		// Convert pgvector back to our Vector type
		vector := make(core.Vec32, len(vec.Slice()))
		copy(vector, vec.Slice())

		// Calculate similarity score (1 - normalized distance)
		// Note: This assumes L2 distance and may need adjustment based on your distance metric
		score := 1.0 - distance

		results = append(results, core.SearchResult{
			ID:      id,
			Score:   score,
			Vector:  vector,
			Content: content,
		})
	}

	if err := rows.Err(); err != nil {
		return nil, err
	}

	return results, nil
}

// Close releases the pgx connection pool resources
func (s *PgVectorStore) Close() {
	if s.pool != nil {
		s.pool.Close()
	}
}
