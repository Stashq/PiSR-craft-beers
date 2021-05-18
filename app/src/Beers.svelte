<script lang="ts">
	
	import BeerList from './Beer_image_list.svelte'
	
	
</script>

	<div style="min-width: 100px; width:100%">
			{#each Array(3) as _unused, i}
			  <BeerList></BeerList>
			{/each}
	  </div>

<style>
	.beer_img{
		width: 50%;
		height: 70%;
	}
</style>