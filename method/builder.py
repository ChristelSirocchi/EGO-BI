import numpy as np
    
class BaseNeighborhoodBuilder:
    def __init__(self, agg_type='mean', filter=False):
        """
        Parameters:
        -----------
        agg_type : str ('mean', 'max', or 'weighted')
        """
        self.agg_type = agg_type
        self.filter = filter

        if agg_type not in ['mean', 'max', 'weighted']:
            raise ValueError("agg_type must be either 'mean', 'max', or 'weighted'")

    def fit_train(self, X, y, D, k, train_ids):
        # Sort distances and get indices of k-nearest neighbors
        knn_idx = np.argsort(D, axis=1)[:, 1:k+1]
        # Gather the corresponding sorted distances
        knn_dists = np.take_along_axis(D, knn_idx, axis=1)
        return self._build(X, y, knn_idx, knn_dists, X, y, train_ids)

    def transform_test(self, X_test, y_test, X_train, y_train, D, k, test_ids):
        knn_idx = np.argsort(D, axis=1)[:, :k]
        knn_dists = np.take_along_axis(D, knn_idx, axis=1)
        return self._build(X_test, y_test, knn_idx, knn_dists, X_train, y_train, test_ids)
    
    def _compute_entropy(self, p):
        if p == 0 or p == 1:
            return 0.0
        else:
            return (-p * np.log2(p) - (1 - p) * np.log2(1 - p))

    def _build(self, X_query, y_query, knn_idx, knn_dists, X_ref, y_ref, query_ids):
        X_out, y_out, z_out, id_out, w_out = [], [], [], [], []

        for i in range(X_query.shape[0]):

            neigh_labels = y_ref[knn_idx[i]]

            feats = self._compute_features(
                center_x=X_query[i],
                center_label=y_query[i],
                neigh_x=X_ref[knn_idx[i]],
                neigh_labels=neigh_labels,
                neigh_dists=knn_dists[i]  # Passed down to compute weights
            )

            #p = np.mean(neigh_labels != y_query[i])
            #entropy = self._compute_entropy(p)

            n_opposite = np.sum(neigh_labels != y_query[i])
            weight = min(n_opposite, len(neigh_labels) - n_opposite) + 1

            for feat, mode, label in feats:
                X_out.append(feat)
                y_out.append(mode)
                z_out.append(label)
                id_out.append(query_ids[i])
                w_out.append(weight)
            
        return np.array(X_out), np.array(y_out), np.array(z_out), np.array(id_out), np.array(w_out)


    def _aggregate(self, diffs, dists, max_missing_ratio=0.5):
        """
        Helper method to perform aggregation based on configured strategy.
        Handles mean, max, and inverse-distance weighted mean.
        """
        n_neighbors = diffs.shape[0]
        nan_mask = np.isnan(diffs)
        missing_counts = np.sum(nan_mask, axis=0)
        missing_ratios = missing_counts / n_neighbors
        too_many_nans_mask = missing_ratios >= max_missing_ratio

        if self.agg_type == 'mean':
            aggregated = np.nanmean(diffs, axis=0)
        elif self.agg_type == 'max':
            aggregated = np.nanmax(diffs, axis=0)
        elif self.agg_type == 'weighted':
            # Compute inverse distance weights (add epsilon to avoid division by zero)
            weights = 1.0 / (dists + 1e-5)
            
            # Mask out any NaN values present in diffs
            nan_mask = np.isnan(diffs)
            if np.any(nan_mask):
                # If NaNs exist, zero out their weights for the dot product
                weights_expanded = np.repeat(weights[:, np.newaxis], diffs.shape[1], axis=1)
                weights_expanded[nan_mask] = 0.0
                
                # Weighted average: sum(diffs * weights) / sum(weights)
                numerator = np.nansum(diffs * weights_expanded, axis=0)
                denominator = np.sum(weights_expanded, axis=0)
                # Avoid division by zero if all weights are 0 for a feature
                denominator = np.where(denominator == 0, 1, denominator)
                aggregated = numerator / denominator
            else:
                # Optimized path if no NaNs are present
                aggregated = np.average(diffs, axis=0, weights=weights)
        if self.filter:
            aggregated[too_many_nans_mask] = np.nan
        return aggregated
    
    def _compute_features(self, center_x, center_label, neigh_x, neigh_labels, neigh_dists):
        raise NotImplementedError

class EntropyBuilder(BaseNeighborhoodBuilder):

    def _compute_features(self, center_x, center_label, neigh_x, neigh_labels, neigh_dists):

        # -------------------------
        # 1. variability per neighbour
        # -------------------------
        # shape: (k, n_features)
        diffs = np.abs(neigh_x - center_x)

        # aggregate variability across neighbors
        feat = self._aggregate(diffs, neigh_dists)

        # -------------------------
        # 2. neighborhood entropy (label distribution)
        # -------------------------
        p = np.mean(neigh_labels != center_label)

        entropy = self._compute_entropy(p)

        return [
            (feat, entropy, center_label)
        ]
      
class SameBuilder(BaseNeighborhoodBuilder):

    def _compute_features(self, center_x, center_label, neigh_x, neigh_labels, neigh_dists):
        opp_mask = (neigh_labels != center_label)
        is_mixed = int(np.any(opp_mask))

        # Compares all samples regardless of label strategy
        diffs = np.abs(neigh_x - center_x)
        feat = self._aggregate(diffs, neigh_dists)

        return [(feat, is_mixed, -1 if is_mixed else center_label)]

       
class SubsetBuilder(BaseNeighborhoodBuilder):

    def _compute_features(self, center_x, center_label, neigh_x, neigh_labels, neigh_dists):
        opp_mask = (neigh_labels != center_label)
        same_mask = ~opp_mask
        is_mixed = int(np.any(opp_mask))

        mask = opp_mask if np.any(opp_mask) else same_mask
        
        if np.any(mask):
            diffs = np.abs(neigh_x[mask] - center_x)
            feat = self._aggregate(diffs, neigh_dists[mask])
        else:
            feat = np.full(center_x.shape, np.nan)

        return [(feat, is_mixed, -1 if is_mixed else center_label)]    
    
    
class SplitBuilder(BaseNeighborhoodBuilder):

    def _compute_features(self, center_x, center_label, neigh_x, neigh_labels, neigh_dists):
        same_mask = (neigh_labels == center_label)
        opp_mask = ~same_mask
        outputs = []

        if np.any(same_mask):
            diffs_same = np.abs(neigh_x[same_mask] - center_x)
            feat_same = self._aggregate(diffs_same, neigh_dists[same_mask])
            outputs.append((feat_same, 0, center_label))

        if np.any(opp_mask):
            diffs_opp = np.abs(neigh_x[opp_mask] - center_x)
            feat_opp = self._aggregate(diffs_opp, neigh_dists[opp_mask])
            outputs.append((feat_opp, 1, -1))

        return outputs


class SplitOnlyBuilder(BaseNeighborhoodBuilder):

    def _compute_features(self, center_x, center_label, neigh_x, neigh_labels, neigh_dists):
        same_mask = (neigh_labels == center_label)
        opp_mask = ~same_mask
        outputs = []

        if np.any(same_mask) and np.any(opp_mask):
            diffs_same = np.abs(neigh_x[same_mask] - center_x)
            feat_same = self._aggregate(diffs_same, neigh_dists[same_mask])
            outputs.append((feat_same, 0, center_label))

            diffs_opp = np.abs(neigh_x[opp_mask] - center_x)
            feat_opp = self._aggregate(diffs_opp, neigh_dists[opp_mask])
            outputs.append((feat_opp, 1, center_label))

        return outputs
    